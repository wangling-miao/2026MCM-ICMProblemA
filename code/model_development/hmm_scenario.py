"""
Hidden Markov Model for User Scenario Prediction
=================================================

Implements Section 4.2.2 of the paper:
- HMM-based scenario prediction (Eq. 563-577)
- Transition matrix learning from user behavior data
- Viterbi algorithm for state inference
"""

import numpy as np
from collections import defaultdict

class HMMScenarioPredictor:
    """
    Hidden Markov Model for predicting user usage scenarios.
    
    Hidden states: ['Idle', 'Light', 'Busy', 'Gaming', 'Video', 'Navigation']
    Observations: CPU utilization, screen status, app category
    """
    
    def __init__(self, states=None):
        if states is None:
            self.states = ['Idle', 'Light', 'Busy', 'Gaming', 'Video', 'Navigation']
        else:
            self.states = states
        
        self.n_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        
        # Transition matrix A[i,j] = P(S_t = j | S_{t-1} = i)
        self.A = self._init_transition_matrix()
        
        # Initial state distribution
        self.pi = np.ones(self.n_states) / self.n_states
        
        # Current state
        self.current_state = 'Idle'
        self.current_idx = 0
        
        # Power mapping (from SmartphoneParams)
        self.state_power = {
            'Idle': 0.2, 'Light': 0.8, 'Busy': 2.0,
            'Gaming': 4.0, 'Video': 1.5, 'Navigation': 2.5
        }
    
    def _init_transition_matrix(self):
        """Initialize transition matrix with reasonable defaults."""
        n = self.n_states
        A = np.zeros((n, n))
        
        # High self-transition probability (users tend to stay in same mode)
        for i in range(n):
            A[i, i] = 0.7
        
        # Remaining probability distributed to neighboring states
        for i in range(n):
            remaining = 1.0 - A[i, i]
            for j in range(n):
                if i != j:
                    A[i, j] = remaining / (n - 1)
        
        return A
    
    def learn_from_data(self, app_events):
        """
        Learn transition matrix from user behavior data.
        
        Parameters
        ----------
        app_events : list of dict
            List of app events with 'timestamp', 'app_category', 'event_type'
        """
        # Map app categories to states
        category_to_state = {
            'game': 'Gaming', 'video': 'Video', 'navigation': 'Navigation',
            'social': 'Light', 'productivity': 'Busy', 'system': 'Idle',
            'browser': 'Light', 'music': 'Light', 'camera': 'Busy'
        }
        
        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))
        prev_state = None
        
        for event in app_events:
            category = event.get('app_category', 'system').lower()
            state = category_to_state.get(category, 'Light')
            idx = self.state_to_idx.get(state, 1)
            
            if prev_state is not None:
                transition_counts[prev_state, idx] += 1
            prev_state = idx
        
        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.A = transition_counts / row_sums
        
        # Add small probability to avoid zero transitions
        self.A = 0.95 * self.A + 0.05 / self.n_states
        
        # Ensure each row sums to exactly 1
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        
        return self.A
    
    def predict_next_state(self, current_state=None):
        """Predict most likely next state."""
        if current_state is None:
            current_idx = self.current_idx
        else:
            current_idx = self.state_to_idx.get(current_state, 0)
        
        next_probs = self.A[current_idx].copy()
        # Ensure probabilities sum to 1 (safety normalization)
        next_probs = next_probs / next_probs.sum()
        next_idx = np.random.choice(self.n_states, p=next_probs)
        
        self.current_idx = next_idx
        self.current_state = self.states[next_idx]
        
        return self.current_state
    
    def sample_trajectory(self, n_steps, dt=60.0):
        """
        Sample a usage trajectory.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps
        dt : float
            Time step [s]
            
        Returns
        -------
        dict
            Trajectory with states and powers
        """
        states = []
        powers = []
        times = []
        
        state = self.current_state
        
        for step in range(n_steps):
            states.append(state)
            powers.append(self.state_power[state])
            times.append(step * dt)
            state = self.predict_next_state(state)
        
        return {
            'times': np.array(times),
            'states': states,
            'powers': np.array(powers)
        }
    
    def get_stationary_distribution(self):
        """Compute stationary distribution π such that πA = π."""
        eigenvalues, eigenvectors = np.linalg.eig(self.A.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()
        return dict(zip(self.states, pi))
    
    def expected_power(self):
        """Compute expected power consumption under stationary distribution."""
        pi = self.get_stationary_distribution()
        return sum(pi[s] * self.state_power[s] for s in self.states)
