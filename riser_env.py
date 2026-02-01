import gymnasium as gym
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

class RiserEnv(gym.Env):
    """
    RISER Gymnasium Environment for real-time steering of LLMs.
    This environment wraps a HuggingFace model and allows token-by-token stepping
    with manual KV-cache management and residual stream intervention.
    """
    def __init__(
        self, 
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        target_layer: int = 15,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_steps: int = 50
    ):
        super(RiserEnv, self).__init__()
        
        self.device = torch.device(device)
        self.target_layer = target_layer
        self.max_steps = max_steps
        
        # Load Model and Tokenizer
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if "cuda" in device else torch.float32
        ).to(self.device)
        self.model.eval()
        
        # Determine d_model from config
        self.d_model = self.model.config.hidden_size
        
        # Observation Space (O_t): Semantic State at layer L-1
        # Shape: (d_model,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.d_model,), dtype=np.float32
        )
        
        # Action Space (A_t): Steering vector to add to the residual stream
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.d_model,), dtype=np.float32
        )
        
        # Hooks and State
        self.hook_handle = None
        self.current_steering_vector = None
        self.captured_hidden_state = None
        self.past_key_values = None
        self.input_ids = None
        self.current_step = 0
        
        # Register the hook
        self._register_hook()

    def _register_hook(self):
        """
        Registers a forward hook on the target layer's residual stream.
        """
        # For Llama-3/TinyLlama architecture, hidden states are in model.model.layers
        layer = self.model.model.layers[self.target_layer]
        
        def hook_fn(module, input, output):
            # output is typically a tuple (hidden_states, optional_caches)
            # We want to modify hidden_states: shape (batch, seq_len, d_model)
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Capture the current hidden state (before steering) for the next observation
            # We take the last token's hidden state
            self.captured_hidden_state = hidden_states[:, -1, :].detach().clone()
            
            # Apply Steering Intervention if action is provided
            if self.current_steering_vector is not None:
                # Add steering vector to the residual stream
                # hidden_states = hidden_states + steering_vector
                hidden_states += self.current_steering_vector.to(hidden_states.dtype)
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        self.hook_handle = layer.register_forward_hook(hook_fn)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        prompt = options.get("prompt", "The future of AI is") if options else "The future of AI is"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.input_ids = inputs.input_ids
        
        self.current_step = 0
        self.past_key_values = None
        self.current_steering_vector = None
        
        # Initial forward pass to populate KV-cache and get initial observation
        with torch.no_grad():
            outputs = self.model(
                self.input_ids, 
                use_cache=True, 
                past_key_values=None
            )
            self.past_key_values = outputs.past_key_values
            
        # The hook already captured the hidden state during the forward pass
        observation = self.captured_hidden_state.cpu().numpy().flatten().astype(np.float32)
        
        return observation, {}

    def step(self, action: np.ndarray):
        """
        Stepping through the generation token-by-token.
        """
        self.current_step += 1
        
        # 1. Register the steering vector from the action
        self.current_steering_vector = torch.from_numpy(action).to(self.device).reshape(1, 1, -1)
        
        # 2. Perform a forward pass for a single token using past_key_values
        # Only use the last token for the next input
        next_input_id = self.input_ids[:, -1:]
        
        with torch.no_grad():
            outputs = self.model(
                next_input_id,
                past_key_values=self.past_key_values,
                use_cache=True
            )
            
            # Update KV cache and input_ids
            self.past_key_values = outputs.past_key_values
            
            # Sample next token (greedy for now)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            self.input_ids = torch.cat([self.input_ids, next_token_id], dim=-1)
            
        # 3. Calculate Observation (captured in hook)
        observation = self.captured_hidden_state.cpu().numpy().flatten().astype(np.float32)
        
        # 4. Calculate Dummy Reward (Placeholder: Safety + Coherence - Cost)
        # In Phase 1, we just provide a placeholder
        reward = 1.0 # Base reward for continuing
        
        # 5. Check Termination
        terminated = False
        if next_token_id.item() == self.tokenizer.eos_token_id or self.current_step >= self.max_steps:
            terminated = True
            
        truncated = False
        info = {
            "token_id": next_token_id.item(),
            "decoded_token": self.tokenizer.decode(next_token_id.item()),
            "full_text": self.tokenizer.decode(self.input_ids[0])
        }
        
        return observation, reward, terminated, truncated, info

    def close(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
        super().close()
