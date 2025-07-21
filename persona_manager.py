#!/usr/bin/env python3
"""
Persona Manager for Nash Equilibrium Simulation
Loads and manages personas from the Nemotron dataset
"""

import random
from typing import List, Dict, Optional
from datasets import load_dataset
import logging

class PersonaManager:
    """Manages loading and selection of personas from Nemotron dataset"""
    
    def __init__(self, num_personas: int = 100, seed: Optional[int] = None):
        """
        Initialize the persona manager
        
        Args:
            num_personas: Number of personas to select (default 100)
            seed: Random seed for reproducible selection
        """
        self.num_personas = num_personas
        self.seed = seed
        self.personas = []
        self.logger = logging.getLogger('PersonaManager')
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Load the personas
        self._load_personas()
    
    def _load_personas(self):
        """Load personas from the Nemotron dataset"""
        try:
            self.logger.info("Loading Nemotron personas dataset...")
            dataset = load_dataset("nvidia/Nemotron-Personas")
            
            # Get total number of available personas
            total_personas = len(dataset['train'])
            self.logger.info(f"Found {total_personas} total personas in dataset")
            
            # Randomly select the requested number of personas
            indices = random.sample(range(total_personas), min(self.num_personas, total_personas))
            
            # Extract selected personas
            for i, idx in enumerate(indices):
                persona_data = dataset['train'][idx]
                # Add an internal ID for tracking
                persona_data['participant_id'] = i
                self.personas.append(persona_data)
            
            self.logger.info(f"Selected {len(self.personas)} personas for simulation")
            
        except Exception as e:
            self.logger.error(f"Error loading personas: {e}")
            raise
    
    def get_persona(self, participant_id: int) -> Dict:
        """
        Get a specific persona by participant ID
        
        Args:
            participant_id: The participant's ID
            
        Returns:
            The persona dictionary for that participant
        """
        if 0 <= participant_id < len(self.personas):
            return self.personas[participant_id]
        else:
            raise ValueError(f"Invalid participant_id: {participant_id}")
    
    def get_persona_description(self, participant_id: int) -> str:
        """
        Get a formatted description of a participant's persona
        
        Args:
            participant_id: The participant's ID
            
        Returns:
            A formatted string describing the persona
        """
        persona = self.get_persona(participant_id)
        
        # Create a rich description combining different aspects
        description = f"""
Persona Profile (Participant #{participant_id}):
- Core: {persona.get('persona', 'No description available')}
- Professional: {persona.get('professional_persona', 'Not specified')}
- Age: {persona.get('age', 'Unknown')}, {persona.get('sex', 'Unknown')}, {persona.get('marital_status', 'Unknown').replace('_', ' ')}
- Location: {persona.get('city', 'Unknown')}, {persona.get('state', 'Unknown')}
- Education: {persona.get('education_level', 'Unknown').replace('_', ' ')}
- Occupation: {persona.get('occupation', 'Unknown').replace('_', ' ')}
"""
        
        # Add skills if available
        skills_list = persona.get('skills_and_expertise_list', '')
        if skills_list and skills_list != "None":
            # Parse the string representation of the list
            try:
                import ast
                skills = ast.literal_eval(skills_list)
                if skills:
                    description += f"- Key Skills: {', '.join(skills[:3])}\n"
            except:
                pass
        
        # Add hobbies if available
        hobbies_list = persona.get('hobbies_and_interests_list', '')
        if hobbies_list and hobbies_list != "None":
            try:
                import ast
                hobbies = ast.literal_eval(hobbies_list)
                if hobbies:
                    description += f"- Interests: {', '.join(hobbies[:3])}\n"
            except:
                pass
        
        return description.strip()
    
    def get_persona_trait(self, participant_id: int) -> str:
        """
        Extract a concise personality trait description for prompts
        
        Args:
            participant_id: The participant's ID
            
        Returns:
            A concise trait description
        """
        persona = self.get_persona(participant_id)
        
        # Try to extract key traits from the persona description
        main_persona = persona.get('persona', '')
        
        # Extract adjectives and key characteristics
        if main_persona:
            # Look for descriptive phrases
            import re
            
            # Common patterns for personality traits
            patterns = [
                r'(disciplined|organized|practical|curious|ambitious|cautious|competitive|creative|resilient|meticulous)',
                r'(risk-averse|profit-driven|analytically calculating|opportunistically aggressive|conservatively patient)',
                r'(sociable|reserved|outgoing|introverted|balanced)',
                r'balances (\w+) with (\w+)'
            ]
            
            traits = []
            for pattern in patterns:
                matches = re.findall(pattern, main_persona, re.IGNORECASE)
                if matches:
                    if isinstance(matches[0], tuple):
                        traits.extend(matches[0])
                    else:
                        traits.extend(matches)
            
            if traits:
                # Combine up to 3 traits
                trait_description = " and ".join(traits[:3])
                return trait_description
        
        # Fallback to a generic description based on demographics
        age = persona.get('age', 40)
        occupation = persona.get('occupation', 'worker').replace('_', ' ')
        
        if age < 30:
            return "young and ambitious"
        elif age < 50:
            return "experienced and balanced"
        else:
            return "seasoned and thoughtful"
    
    def get_all_personas(self) -> List[Dict]:
        """Get all loaded personas"""
        return self.personas