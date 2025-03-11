import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from .policy_network import PolicyNetwork
from .state_processor import StateProcessor

class PPOAgentWithFeedback(PPOAgent):
    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        super().__init__(state_dim, action_dim, device)
    
    def request_feedback(self) -> float:
        """Request feedback on response accuracy on a scale of 0 to 5."""
        while True:
            try:
                feedback = float(input("Was the response accurate? Rate from 0 to 5: "))
                if 0 <= feedback <= 5:
                    return feedback
                else:
                    print("Please enter a rating between 0 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 5.")
    
    def update_with_feedback(self, states: List[np.ndarray], actions: List[int],
                             old_probs: List[float], rewards: List[float], feedback: float) -> Dict[str, float]:
        """Update policy using PPO and integrate user feedback into the reward."""
        adjusted_rewards = [reward + feedback for reward in rewards]
        return self.update(states, actions, old_probs, adjusted_rewards)

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action using the policy network."""
        state_tensor = self.state_processor.convert_to_tensor(state, self.device)
        with torch.no_grad():
            probs = self.policy(state_tensor)
        
        action = torch.multinomial(probs, 1).item()
        action_prob = probs[0][action].item()
        
        # Request feedback on the generated response
        feedback = self.request_feedback()  # Get feedback from user
        
        # Update the agent with feedback (integrate into reward)
        self.update_with_feedback([state], [action], [action_prob], [0.0], feedback)  # Assuming initial reward is 0.0
        
        return action, action_prob
    
    def _update_policy(self, states_tensor, actions_tensor, old_probs_tensor, rewards_tensor):
        """Internal policy update logic."""
        for _ in range(5):  # PPO epochs
            probs = self.policy(states_tensor)
            new_probs = probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
            
            ratio = new_probs / old_probs_tensor
            surr1 = ratio * rewards_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * rewards_tensor
            loss = -torch.min(surr1, surr2).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return {'loss': loss.item()}
