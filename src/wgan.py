#@title Base A2C agent. Don't read this if you want to go for the hard exercise at some point. { form-width: "30%" }
import haiku as hk
import chex
import jax
from typing import *
import optax
import jax.numpy as jnp


from .models import GeneratorNetwork, CriticNetwork
from .data import LearnerState

class WGAN:
    '''
    Implement a WGAN, as described in the paper 'tochange'
    
    '''
    def __init__(self, seed: int,
        generator_learning_rate: float, 
        critic_learning_rate: float, 
        n_critic: int,
        clipping_value: float,
        batch_size: int, 
        latent_space_dim: int,
        data_dim: int
        ) -> None:
        """
        :param seed: random seed
        :param generator_learning_rate: lr of generator
        :param critic_learning_rate: lr of generator
        :param n_critic: number of critic iterations per generator iteration
        :param clipping_value: max absolute value of critic params
        :param batch_size: size of batch
        """
        self._rng = jax.random.PRNGKey(seed=seed)
        # hk.transform
        self._init_critic_loss, _apply_critic_loss = hk.without_apply_rng(hk.transform(self._critic_loss_function))
        self._init_generator_loss, _apply_generator_loss = hk.without_apply_rng(hk.transform(self._generator_loss_function))       
       
        self._grad_generator = jax.value_and_grad(_apply_generator_loss, has_aux=True)
        self._grad_critic = jax.value_and_grad(_apply_critic_loss, has_aux=True)

        _, self._apply_critic = hk.without_apply_rng(hk.transform(self._hk_apply_critic))
        _, self._apply_generator = hk.without_apply_rng(hk.transform(self._hk_apply_generator))


        self._n_critic = n_critic
        self._clipping_value = clipping_value
        self._batch_size = batch_size
        self._latent_space_dim =latent_space_dim
        self._data_dim = data_dim

        self._critic_optimizer = optax.rmsprop(learning_rate = critic_learning_rate)
        self._generator_optimizer = optax.rmsprop(learning_rate = - generator_learning_rate)

        self.init_fn = jax.jit(self._init_fn)
        self.update_critic = jax.jit(self._update_critic)
        self.update_generator = jax.jit(self._update_generator)
        self.apply_critic = jax.jit(self._apply_critic)
        self.apply_generator = jax.jit(self._apply_generator)

        self._rng, init_rng = jax.random.split(self._rng)
        self._learner_state = self.init_fn(init_rng, self._generate_dummy_data(), self._generate_dummy_latent_data())

    def _generate_dummy_data(self) -> chex.Array:
        """
        Generate a dummy data for initialization
        :return: dummy data
        """
        return jax.random.normal(self._rng, shape=(self._batch_size, self._data_dim,))

    def _generate_dummy_latent_data(self) -> chex.Array:
        """
        Generate a dummy latent data point for initialization
        :return: dummy data
        """
        return jax.random.normal(self._rng, shape=(self._batch_size, self._latent_space_dim,))

    def _init_fn(self, rng: chex.PRNGKey, data: chex.Array, prior_data:chex.Array) -> LearnerState:
        """
        Initializes the networks and the optimizers
        """
        params = self._init_critic_loss(rng, data, prior_data)

        # get only critic params
        critic_params = {k:params[k] for k in params.keys() if  k.startswith("critic/")}

        # clip them
        def clip(c_data) :
          return jnp.clip(c_data, -self._clipping_value, self._clipping_value)
        new_critic_params_clipped = jax.tree_util.tree_map(clip, critic_params)
        params.update(new_critic_params_clipped)
        # define critic optimizer
        opt_critic_state = self._critic_optimizer.init(new_critic_params_clipped)


        generator_params = {k:params[k] for k in params.keys() if  k.startswith("generator/")}

        opt_generator_state = self._generator_optimizer.init(generator_params)

        return LearnerState(params=params, opt_critic_state=opt_critic_state, opt_generator_state=opt_generator_state)

    def _critic_loss_function(self,data, prior_samples) -> Tuple[chex.Array]:

        generated_data =GeneratorNetwork(data_dim=self._data_dim,name='generator')(prior_samples)
        critic_network =CriticNetwork(name='critic')
        real_values = critic_network(data)
        generated_values = critic_network(generated_data)

        generated_loss = jnp.mean(generated_values)
        real_loss = jnp.mean(real_values)
        loss = real_loss - generated_loss

        logs = dict(generated_loss=generated_loss,
                    real_loss=real_loss,
                    loss=loss)
        return -loss, logs


    def _generator_loss_function(self, prior_samples) -> Tuple[chex.Array, Dict]:
        #critic loss
        generated_data =GeneratorNetwork(self._data_dim, name='generator')(prior_samples)
        generated_values = CriticNetwork(name='critic')(generated_data)
        generated_loss = jnp.mean(generated_values)

        logs = dict(generated_loss=generated_loss)
        
        return generated_loss, logs

    def _update_critic(self, learner_state, data, prior_data):
        (critic_loss, aux_critic), critic_grads = self._grad_critic(learner_state.params, data, prior_data)
  
        # Select the only params and gradients we are interested in for this part, the critics
        critic_params = {k:learner_state.params[k] for k in learner_state.params.keys() if  k.startswith("critic/")}
        critic_grads_only = {k:critic_grads[k] for k in critic_grads.keys() if k.startswith("critic/")}
  
        # Perform one optimization step
        critic_udpates, new_opt_critic_state = self._critic_optimizer.update(critic_grads_only, learner_state.opt_critic_state, critic_params)
        new_critic_params = optax.apply_updates(critic_params, critic_udpates)
 
        def clip(c_data) :
          return jnp.clip(c_data, -self._clipping_value, self._clipping_value)
  
        new_critic_params_clipped = jax.tree_util.tree_map(clip, new_critic_params)
  
        # Update the critic
        learner_state.params.update(new_critic_params_clipped)
  
        return LearnerState(params=learner_state.params, opt_critic_state=new_opt_critic_state, opt_generator_state=learner_state.opt_generator_state), aux_critic

    def _update_generator(self, learner_state, prior_data) -> Tuple[LearnerState, Dict]:
        # generator update
        #   Compute the generator loss and the gradients for ALL the parameters
        (generator_loss, aux_generator), generator_grads = self._grad_generator(learner_state.params, prior_data)

        # Select the only params and gradients we are interested in for this part, the generators
        generator_params = {k:learner_state.params[k] for k in learner_state.params.keys() if  k.startswith("generator/")}
        generator_grads_only = {k:generator_grads[k] for k in generator_grads.keys() if k.startswith("generator/")}

        # Perform one optimization step
        generator_updates, new_opt_generator_state = self._generator_optimizer.update(generator_grads_only, learner_state.opt_generator_state, generator_params)
        new_generator_params = optax.apply_updates(generator_params, generator_updates)

        # Update the generator
        learner_state.params.update(new_generator_params)

        return LearnerState(params=learner_state.params, opt_critic_state=learner_state.opt_critic_state, opt_generator_state=new_opt_generator_state), aux_generator

    def _hk_apply_critic(self, data) -> chex.Array:
        return CriticNetwork(name='critic')(data)

    def _hk_apply_generator(self, latent_data) -> chex.Array:
        return GeneratorNetwork(self._data_dim, name='generator')(latent_data)

    def critic(self, data):
        critic_params = {k:self._learner_state.params[k] for k in self._learner_state.params.keys() if  k.startswith("critic/")}
        return self.apply_critic(critic_params, data)

    def generator(self, latent_data):
        generator_params = {k:self._learner_state.params[k] for k in self._learner_state.params.keys() if  k.startswith("generator/")}
        return self.apply_generator(generator_params, latent_data)