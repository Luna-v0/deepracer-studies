"""TRPO on top of rsl-rl's PPO: only the policy update changes.

Where PPO bounds the step by clipping the objective, TRPO measures the KL
divergence between the old and the new policy and takes the largest step that
fits a fixed KL budget. Collection, advantages, networks, storage and
checkpointing are inherited from PPO unchanged, so the class drops into the
same OnPolicyRunner via `Algo(cls=TRPO)`.

Recurrent policies, RND, symmetry augmentation and actor/critic parameter
sharing are not supported; `update` raises on each of them.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO


def _flat(tensors) -> torch.Tensor:
    """Concatenates tensors into one flat vector."""
    return torch.cat([t.reshape(-1) for t in tensors])


class TRPO(PPO):
    """PPO with a trust-region policy step instead of a clipped objective.

    Settings are class attributes rather than constructor arguments: the `Algo`
    stage records `params` in the spec but never forwards it to the algorithm.

    Attributes:
        max_kl: KL budget allowed per policy update.
        cg_iters: conjugate-gradient iterations.
        cg_damping: damping added to the curvature product.
        line_search_steps: most step halvings tried before giving up.
        critic_epochs: critic optimization epochs per update.
        critic_mini_batches: critic mini-batches per epoch.
    """

    max_kl: float = 0.01
    cg_iters: int = 10
    cg_damping: float = 0.1
    line_search_steps: int = 10
    critic_epochs: int = 5
    critic_mini_batches: int = 4

    def _actor_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.actor.parameters() if p.requires_grad]

    def _get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.reshape(-1) for p in self._actor_params()])

    def _set_flat_params(self, flat: torch.Tensor) -> None:
        offset = 0
        for p in self._actor_params():
            n = p.numel()
            p.data.copy_(flat[offset:offset + n].view_as(p))
            offset += n

    def _evaluate(self, batch):
        """Recomputes the surrogate, the KL to the rollout policy, and entropy.

        Args:
            batch: One rollout batch from the storage generator.

        Returns:
            The surrogate objective to maximize, the mean KL divergence from
            the policy that collected the batch, and the mean entropy.
        """
        self.actor(batch.observations, stochastic_output=True)
        log_prob = self.actor.get_output_log_prob(batch.actions)
        ratio = torch.exp(log_prob - torch.squeeze(batch.old_actions_log_prob))
        surrogate = (torch.squeeze(batch.advantages) * ratio).mean()
        kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, self.actor.output_distribution_params).mean()
        return surrogate, kl, self.actor.output_entropy.mean()

    def _conjugate_gradient(self, matrix_vector_product,
                            b: torch.Tensor) -> torch.Tensor:
        """Solves ``H x = b`` without ever forming H.

        Args:
            matrix_vector_product: Callable returning ``H v`` for a vector v.
            b: Right-hand side of the system.

        Returns:
            The approximate solution x.
        """
        x = torch.zeros_like(b)
        residual = b.clone()
        direction = b.clone()
        rr = residual.dot(residual)
        for _ in range(self.cg_iters):
            hd = matrix_vector_product(direction)
            alpha = rr / (direction.dot(hd) + 1e-8)
            x += alpha * direction
            residual -= alpha * hd
            rr_next = residual.dot(residual)
            if rr_next < 1e-10:
                break
            direction = residual + (rr_next / rr) * direction
            rr = rr_next
        return x

    def update(self) -> dict[str, float]:
        """Takes one trust-region policy step, then fits the critic.

        Returns:
            Scalar losses and step diagnostics for the runner's logger.

        Raises:
            NotImplementedError: For recurrent policies, RND, symmetry, or an
                actor and critic sharing parameters.
        """
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise NotImplementedError("TRPO does not support recurrent policies")
        if self.rnd is not None or self.symmetry is not None:
            raise NotImplementedError("TRPO does not support RND or symmetry")

        params = self._actor_params()
        # Camera policies share their CNN encoder between actor and critic
        # (share_cnn_encoders). The unconstrained critic step below would then
        # move the policy again, right out of the trust region just enforced.
        actor_ids = {id(p) for p in params}
        if any(id(p) in actor_ids for p in self.critic.parameters()):
            raise NotImplementedError(
                "TRPO does not support an actor and critic sharing parameters")

        # One batch over the whole rollout: the natural gradient is defined on
        # the full sample, not per mini-batch as in PPO.
        batch = next(self.storage.mini_batch_generator(num_mini_batches=1,
                                                       num_epochs=1))

        surrogate, kl, entropy = self._evaluate(batch)
        grads = torch.autograd.grad(surrogate, params, retain_graph=True)
        gradient = _flat(grads).detach()

        # Keep the KL gradient's graph so it can be differentiated a second
        # time: that second derivative is the curvature, and contracting it
        # with a vector avoids materializing the full matrix.
        kl_gradient = _flat(torch.autograd.grad(kl, params, create_graph=True))

        def curvature_times(v: torch.Tensor) -> torch.Tensor:
            hv = torch.autograd.grad((kl_gradient * v).sum(), params, retain_graph=True)
            return _flat(hv).detach() + self.cg_damping * v

        direction = self._conjugate_gradient(curvature_times, gradient)
        curvature = direction.dot(curvature_times(direction))
        if curvature <= 0:      # degenerate batch: stay where we are
            full_step = torch.zeros_like(direction)
        else:
            full_step = torch.sqrt(2.0 * self.max_kl / (curvature + 1e-8)) * direction

        # Backtracking line search: accept the first step that improves the
        # surrogate without leaving the trust region.
        start_params = self._get_flat_params()
        surrogate_before = surrogate.item()
        accepted_fraction, accepted_kl, gain = 0.0, 0.0, 0.0
        for i in range(self.line_search_steps):
            fraction = 0.5 ** i
            self._set_flat_params(start_params + fraction * full_step)
            with torch.no_grad():
                new_surrogate, new_kl, _ = self._evaluate(batch)
            if new_kl.item() <= self.max_kl and new_surrogate.item() > surrogate_before:
                accepted_fraction, accepted_kl = fraction, new_kl.item()
                gain = new_surrogate.item() - surrogate_before
                break
        else:
            self._set_flat_params(start_params)

        # The critic is unconstrained; only its parameters get gradients here,
        # so the shared optimizer leaves the actor untouched.
        value_loss_sum, n_updates = 0.0, 0
        for critic_batch in self.storage.mini_batch_generator(
                self.critic_mini_batches, self.critic_epochs):
            values = self.critic(critic_batch.observations)
            value_loss = (critic_batch.returns - values).pow(2).mean()
            self.optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            value_loss_sum += value_loss.item()
            n_updates += 1

        observations = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(observations)
        self.critic.update_normalization(observations)
        self.storage.clear()

        # surrogate_before is ~0 by construction (ratio is 1 and advantages are
        # centered), so the step's gain is what carries information.
        return {
            "value": value_loss_sum / max(n_updates, 1),
            "surrogate_gain": gain,
            "entropy": entropy.item(),
            "kl": accepted_kl,
            "step_fraction": accepted_fraction,
        }
