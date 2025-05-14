"""
Logic for running a single simulation instance.
"""
from simulation.engine import SimulationEngine
from simulation.logger import SimulationLogger
from world.planet import Planet
from agents.agent import Agent
from agents.ml_behavior import MLBehavior
from world.galaxy import Galaxy
from world.system import StarSystem
import random
import logging
from collections import defaultdict
from typing import Dict, List, Tuple
from agents.agent import Action
import joblib
import time
import math

# Import visualization components - with fallback if pygame is not installed
try:
    from visualization.adapter import visualize_simulation
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Pygame not installed - visualization unavailable")

def run_simulation(seed, sim_id):
    """Run the simulation for a specified number of ticks."""
    # Initialize simulation components
    engine = SimulationEngine(seed=seed, ticks=100, name=f"Sim_{sim_id}")
    
    # Create galaxy and system structure
    galaxy = Galaxy(name=f"Galaxy_{sim_id}", seed=seed)
    
    # Generate random coordinates for the system
    system_position = (
        random.randint(0, 1000),
        random.randint(0, 1000),
        random.randint(0, 1000)
    )
    system = StarSystem(
        name=f"System_{sim_id}",
        position=system_position
    )
    galaxy.add_system(system)
    
    # Create test planets with proper spacing
    num_planets = random.randint(3, 7)
    create_planets_with_spacing(system, num_planets, sim_id, seed)
    
    # Get list of habitable planets
    habitable_planets = [p for p in system.planets if p.is_habitable()]
    if not habitable_planets:
        habitable_planets = [system.planets[0]]  # Fallback to first planet
        
    # Create agents and distribute them across habitable planets
    agents = []
    for i in range(10):  # Create 10 agents
        planet = random.choice(habitable_planets)
        agent = Agent(f"Agent_{i}", planet)
        agent.behavior = MLBehavior()  # Add ML-based behavior
        agents.append(agent)
        planet.add_agent(agent)
        
    # Initialize logger
    json_logger = SimulationLogger(str(sim_id))
    
    # Run simulation ticks
    for tick in range(100):  # Run for 100 ticks
        print(f"\nTick {tick}:")
        print(f"Alive agents: {len([a for a in agents if a.alive])}")
        
        # Update planet state
        for planet in system.planets:
            planet.update()
            
        # Print planet state
        print("Planet state:")
        for planet in system.planets:
            print(f"\nPlanet {planet.name}:")
            print(f"  Temperature: {planet.temperature}")
            print(f"  Weather: {planet.weather}")
            print(f"  Hazards: {planet.hazards}")
            print(f"  Resources: {planet.resources}")
            print(f"  Agents: {len(planet.agents)}")
        
        # Update and print agent states
        for agent in agents:
            if not agent.alive:
                continue
                
            print(f"\nAgent {agent.id} state:")
            print(f"  Energy: {agent.energy}")
            print(f"  Health: {agent.health}")
            print(f"  Inventory: {agent.inventory}")
            print(f"  Planet: {agent.planet.name}")
            
            # Get action from ML behavior
            action = agent.behavior.decide(agent)
            
            if action:
                print(f"  Action: {action.name}")
                success = agent.execute(action)
                print(f"  Success: {success}")
                print(f"  Decision reason: {agent.behavior.get_decision_reason()}")
                print(f"  New energy: {agent.energy}")
                print(f"  New health: {agent.health}")
                print(f"  New inventory: {agent.inventory}")
                
            # Apply environmental effects
            agent._apply_environmental_effects()
            print("  After environment:")
            print(f"    Energy: {agent.energy}")
            print(f"    Health: {agent.health}")
        
        # Log the current state for each planet
        for planet in system.planets:
            if planet.agents:
                json_logger.log_tick(planet, planet.agents)
        
        # Check if all agents are dead
        if all(not agent.alive for agent in agents):
            print(f"All agents died at tick {tick}")
            break
    
    return json_logger

def get_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Calculate the euclidean distance between two points in 3D space."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def create_planets_with_spacing(system: StarSystem, num_planets: int, sim_id: int, seed: int):
    """Create planets for the system with minimum spacing between them."""
    # Set a random seed for deterministic planet placement
    planet_random = random.Random(seed)
    
    # Minimum distance between planets for visualization - increased to prevent overlap
    MIN_PLANET_DISTANCE = 1000  # Increased from 800 to further spread planets
    
    # Radius range that planets will be placed from the system center
    MIN_ORBIT_RADIUS = 600  # Minimum orbit radius
    MAX_ORBIT_RADIUS = 2000  # Maximum orbit radius
    
    # Explicitly enforce different quadrants for planets to prevent overlap
    quadrants = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # (x sign, y sign)
    
    # Create orbit "bands" with precise fixed radii to ensure good separation
    orbit_bands = []
    # Use fixed radii instead of ranges to ensure clear separation
    fixed_radii = [
        MIN_ORBIT_RADIUS,
        MIN_ORBIT_RADIUS + (MAX_ORBIT_RADIUS - MIN_ORBIT_RADIUS) * 0.25,
        MIN_ORBIT_RADIUS + (MAX_ORBIT_RADIUS - MIN_ORBIT_RADIUS) * 0.5,
        MIN_ORBIT_RADIUS + (MAX_ORBIT_RADIUS - MIN_ORBIT_RADIUS) * 0.75,
        MAX_ORBIT_RADIUS
    ]
    
    # Calculate number of bands based on number of planets, but no more than 5
    num_bands = min(num_planets, 5)
    
    # Create bands with fixed radii
    for i in range(num_bands):
        radius = fixed_radii[i]
        # Define each band with an exact radius instead of a range
        orbit_bands.append(radius)
    
    # List to keep track of placed planets coordinates
    planet_positions = []
    all_planets = []
    
    # Assign planets to bands in a balanced way
    planets_per_band = {}
    remaining_planets = num_planets
    
    # Distribute planets evenly across bands
    base_planets_per_band = num_planets // num_bands
    extra_planets = num_planets % num_bands
    
    for band_idx in range(num_bands):
        # Add an extra planet to early bands if we have extras
        planets_per_band[band_idx] = base_planets_per_band + (1 if band_idx < extra_planets else 0)
        
    # Create planets for each orbit band with exact positioning
    planet_idx = 0
    for band_idx, radius in enumerate(orbit_bands):
        num_band_planets = planets_per_band.get(band_idx, 0)
        
        # Skip if no planets in this band
        if num_band_planets <= 0:
            continue
        
        # Calculate exact angle spacing for this band
        angle_step = 2 * math.pi / num_band_planets
        
        # Assign quadrants to ensure planets are well distributed
        band_quadrants = []
        for q in quadrants:
            band_quadrants.extend([q] * ((num_band_planets + 3) // 4))  # Ensure we have enough quadrants
        band_quadrants = band_quadrants[:num_band_planets]  # Trim to exactly what we need
        # Shuffle quadrants for this band
        planet_random.shuffle(band_quadrants)
        
        for i in range(num_band_planets):
            # Index for this planet
            if planet_idx >= num_planets:
                break
                
            # Exact angle based on position in the band without any variation
            base_angle = i * angle_step
            
            # Apply quadrant adjustment if needed
            quadrant_x, quadrant_y = band_quadrants[i]
            
            # Calculate exact position - no random variation to ensure consistent spacing
            x = radius * math.cos(base_angle) * quadrant_x
            y = radius * math.sin(base_angle) * quadrant_y
            z = planet_random.randint(-20, 20)  # Minimal z-axis variation
            pos = (x, y, z)
            
            # Check distance from all existing planets
            position_is_valid = all(
                get_distance(pos, existing_pos) >= MIN_PLANET_DISTANCE 
                for existing_pos in planet_positions
            )
            
            # If position is valid or this is the first planet, use it
            if position_is_valid or not planet_positions:
                planet_positions.append(pos)
            else:
                # Try alternate placement with increasing radius until we find a valid position
                for radius_multiplier in range(2, 5):
                    for angle_offset in [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]:
                        new_angle = base_angle + angle_offset
                        new_radius = radius * radius_multiplier
                        
                        x = new_radius * math.cos(new_angle) * quadrant_x
                        y = new_radius * math.sin(new_angle) * quadrant_y
                        z = planet_random.randint(-20, 20)
                        pos = (x, y, z)
                        
                        position_is_valid = all(
                            get_distance(pos, existing_pos) >= MIN_PLANET_DISTANCE 
                            for existing_pos in planet_positions
                        )
                        
                        if position_is_valid:
                            planet_positions.append(pos)
                            break
                    
                    if len(planet_positions) > len(all_planets):
                        # We added a position, so break out of the radius loop
                        break
                
                # If still no valid position, place it very far away
                if len(planet_positions) <= len(all_planets):
                    far_angle = planet_random.uniform(0, 2 * math.pi)
                    far_radius = MAX_ORBIT_RADIUS * 3.0
                    x = far_radius * math.cos(far_angle)
                    y = far_radius * math.sin(far_angle)
                    z = planet_random.randint(-20, 20)
                    planet_positions.append((x, y, z))
            
            # Create planet with the determined position
            planet = Planet(
                name=f"Planet_{sim_id}_{planet_idx}",
                seed=seed + planet_idx
            )
            
            # Set coordinates
            planet.coords = planet_positions[-1]
            
            # Add to list of all planets
            all_planets.append(planet)
            
            # Next planet
            planet_idx += 1
    
    # Make a final pass to ensure no planets are too close together
    fix_too_close_planets(all_planets, MIN_PLANET_DISTANCE)
    
    # Add planets to system
    for planet in all_planets:
        system.add_planet(planet)

def fix_too_close_planets(planets, min_distance):
    """Apply additional fixes to ensure all planets maintain minimum distance."""
    # Make multiple passes to handle complex cases
    MAX_PASSES = 8  # Increased from 5 to 8 passes for more thorough fixing
    for pass_num in range(MAX_PASSES):
        need_fixes = False
        
        # Check all planet pairs
        for i, planet1 in enumerate(planets):
            for j, planet2 in enumerate(planets):
                if i >= j:  # Skip same planet and already checked pairs
                    continue
                
                p1_pos = planet1.coords
                p2_pos = planet2.coords
                dist = get_distance(p1_pos, p2_pos)
                
                if dist < min_distance:
                    need_fixes = True
                    
                    # Calculate direction from p1 to p2
                    direction_x = p2_pos[0] - p1_pos[0]
                    direction_y = p2_pos[1] - p1_pos[1]
                    direction_z = p2_pos[2] - p1_pos[2]
                    
                    # Normalize direction
                    direction_length = math.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
                    if direction_length > 0:
                        direction_x /= direction_length
                        direction_y /= direction_length
                        direction_z /= direction_length
                    
                    # Move both planets away from each other
                    # More aggressive movement for closer planets
                    proximity_factor = 2.0 + (min_distance - dist) / min_distance * 3.0
                    move_distance = (min_distance - dist) / 2 + 150 * proximity_factor
                    
                    # Move planet1 away from planet2
                    new_p1_x = p1_pos[0] - direction_x * move_distance
                    new_p1_y = p1_pos[1] - direction_y * move_distance
                    new_p1_z = p1_pos[2] - direction_z * move_distance
                    
                    # Move planet2 away from planet1
                    new_p2_x = p2_pos[0] + direction_x * move_distance
                    new_p2_y = p2_pos[1] + direction_y * move_distance
                    new_p2_z = p2_pos[2] + direction_z * move_distance
                    
                    # Update positions
                    planet1.coords = (new_p1_x, new_p1_y, new_p1_z)
                    planet2.coords = (new_p2_x, new_p2_y, new_p2_z)
        
        # If no planets needed fixes, we can stop
        if not need_fixes:
            break
    
    # Final check for any remaining clustered planets
    # Apply extreme separation for any remaining problem cases
    for i, planet1 in enumerate(planets):
        for j, planet2 in enumerate(planets):
            if i >= j:
                continue
                
            dist = get_distance(planet1.coords, planet2.coords)
            if dist < min_distance:
                # Apply extreme separation - move planets to opposite sides
                angle1 = random.uniform(0, 2 * math.pi)
                angle2 = angle1 + math.pi  # Opposite direction
                
                radius = min_distance * 3
                
                # Place planets at these extreme positions
                planet1.coords = (
                    radius * math.cos(angle1),
                    radius * math.sin(angle1),
                    random.randint(-20, 20)
                )
                
                planet2.coords = (
                    radius * math.cos(angle2),
                    radius * math.sin(angle2),
                    random.randint(-20, 20)
                )
    
    # Final cluster breaking - if there are still pairs too close
    close_planets = []
    for i, planet1 in enumerate(planets):
        close_count = 0
        for j, planet2 in enumerate(planets):
            if i == j:
                continue
                
            dist = get_distance(planet1.coords, planet2.coords)
            if dist < min_distance * 1.2:  # Check for planets that are still a bit too close
                close_count += 1
        
        if close_count >= 2:  # If a planet is close to multiple others
            close_planets.append(i)
    
    # Move clustered planets to extreme positions in different directions
    if close_planets:
        # Add safety fix - move clusters to extreme corners
        num_quadrants = 4
        for i, planet_idx in enumerate(close_planets):
            quadrant = i % num_quadrants
            angle = quadrant * (math.pi / 2) + random.uniform(0, math.pi / 4)
            
            # Use an extreme radius
            radius = min_distance * 4
            
            # Move planet to this extreme position
            planets[planet_idx].coords = (
                radius * math.cos(angle),
                radius * math.sin(angle),
                random.randint(-20, 20)
            )

class SimulationRunner:
    def __init__(self, debug=False, ticks_per_simulation=100):
        self.debug = debug
        self.ticks_per_simulation = ticks_per_simulation
        self.current_sim_id = 0
        self.action_counts = defaultdict(int)
    
    def run(self):
        """Run a new simulation."""
        seed = random.randint(1000, 10000)
        sim_id = self.current_sim_id
        self.current_sim_id += 1
        
        print(f"Starting simulation {sim_id} with seed {seed}")
        
        # Load ML model
        try:
            model = joblib.load("models/agent_behavior_model.pkl")
            print("Successfully loaded model from models/agent_behavior_model.pkl")
        except:
            print("No ML model found, using default behavior")
            model = None
            
        return self.run_simulation(seed, sim_id)
        
    def simulate(self, seed=None, sim_id=None, ticks=100, output_interval=10, visualize=False):
        """Run simulation with minimal output for performance.
        
        Args:
            seed: Random seed for simulation
            sim_id: Simulation ID (will use counter if None)
            ticks: Number of ticks to simulate
            output_interval: How often to print status (0 for no output)
            visualize: Whether to show visualization
        """
        if seed is None:
            seed = random.randint(1000, 10000)
        if sim_id is None:
            sim_id = self.current_sim_id
            self.current_sim_id += 1
        
        print(f"Starting fast simulation {sim_id} with seed {seed}")
        
        # Create simulation components with minimal initialization
        galaxy = Galaxy(name=f"Galaxy_{sim_id}", seed=seed)
        system = StarSystem(
            name=f"System_{sim_id}",
            position=(random.randint(0, 1000), random.randint(0, 1000), random.randint(0, 1000))
        )
        galaxy.add_system(system)
        
        # Create planets with proper spacing
        num_planets = random.randint(3, 7)
        create_planets_with_spacing(system, num_planets, sim_id, seed)
        
        # Distribute agents across planets
        agents = []
        for i in range(10):
            planet = random.choice(system.planets)
            agent = Agent(f"Agent_{i}", planet)
            agent.behavior = MLBehavior()
            agents.append(agent)
            planet.add_agent(agent)
        
        # Setup JSON logger
        json_logger = SimulationLogger(str(sim_id), seed, galaxy)
        
        # Variables to track resource depletion
        resource_depletion_tick = ticks // 3
        resource_depletion_done = False
        migration_candidates = []
        
        # Initialize visualization if requested
        visualization_adapter = None
        if visualize and VISUALIZATION_AVAILABLE:
            visualization_adapter = visualize_simulation(galaxy)
        
        # Run simulation ticks with minimal output
        running = True
        tick = 0
        
        while running and tick < ticks:
            # Process visualization events if visualization is active
            if visualization_adapter:
                # Update current tick in visualization
                visualization_adapter.update_tick(tick)
                
                # Process events and check if we should quit
                if not visualization_adapter.render_and_process_events():
                    print("Visualization closed, stopping simulation.")
                    break
                
                # Check if visualization is paused - if so, just render and continue loop
                if visualization_adapter.is_paused():
                    time.sleep(0.05)  # Small delay to prevent high CPU usage
                    continue
                
            # Output status at specified intervals
            if output_interval > 0 and tick % output_interval == 0:
                alive_agents = sum(1 for agent in agents if agent.alive)
                print(f"Tick {tick}/{ticks} - Alive agents: {alive_agents}")
                
                # Print resource status for planets with agents
                for planet in system.planets:
                    if planet.agents:
                        food = planet.resources.get('food', 0)
                        water = planet.resources.get('water', 0)
                        print(f"  {planet.name}: {len(planet.agents)} agents, food={food}, water={water}")
            
            # Force resource depletion to test migration
            if tick == resource_depletion_tick and not resource_depletion_done:
                # Find planets with agents to deplete
                planets_with_agents = [p for p in system.planets if p.agents]
                if planets_with_agents:
                    target_planet = planets_with_agents[0]  # Deplete first planet with agents
                    target_planet.deplete_resources(amount=800)
                    print(f"DEPLETED RESOURCES on {target_planet.name} to trigger migration")
                    migration_candidates = [a.id for a in target_planet.agents]
                    resource_depletion_done = True
            
            # Process agents
            for agent in agents:
                if agent.alive:
                    action = agent.behavior.decide(agent)
                    if action and agent._can_perform_action(action):
                        success = agent.execute(action)
                        if action and success:
                            self.action_counts[action.name] = self.action_counts.get(action.name, 0) + 1
                            # Track migrations
                            if action.name == "MIGRATE" and agent.id in migration_candidates:
                                print(f"MIGRATION: Agent {agent.id} is migrating from {agent.planet.name}")
                    agent._apply_environmental_effects()
            
            # Update planets
            for planet in system.planets:
                planet.tick()
            
            # Log only at specified intervals for efficiency
            if output_interval > 0 and tick % output_interval == 0:
                for planet in system.planets:
                    if planet.agents:
                        json_logger.log_tick(planet, planet.agents)
            
            # Check if all agents are dead
            if all(not agent.alive for agent in agents):
                print(f"All agents died at tick {tick}")
                break
            
            # Increment tick counter
            tick += 1
            
            # Add small delay to make visualization smooth if enabled
            if visualization_adapter and not visualization_adapter.is_paused():
                time.sleep(0.05)  # 50ms delay for smoother visualization
        
        # Final log regardless of interval
        for planet in system.planets:
            if planet.agents:
                json_logger.log_tick(planet, planet.agents)
                
        print(f"Fast simulation {sim_id} completed")
        
        # Print summary
        alive_agents = sum(1 for agent in agents if agent.alive)
        print(f"Final state: {alive_agents}/{len(agents)} agents alive")
        
        # Show migration results
        print("\nMigration check:")
        for planet in system.planets:
            agent_ids = [a.id for a in planet.agents]
            print(f"  {planet.name}: {len(planet.agents)} agents - {agent_ids}")
        
        # Show action statistics
        if self.action_counts:
            print("\nAction counts:")
            for action, count in sorted(self.action_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {action}: {count}")
        
        # Clean up visualization if active
        if visualization_adapter:
            visualization_adapter.stop()
        
        return json_logger
        
    def run_simulation(self, seed, sim_id):
        """Run the simulation for a specified number of ticks."""
        # Initialize simulation components
        engine = SimulationEngine(seed=seed, ticks=self.ticks_per_simulation, name=f"Sim_{sim_id}")
        
        # Create galaxy and system structure
        galaxy = Galaxy(name=f"Galaxy_{sim_id}", seed=seed)
        
        # Generate random coordinates for the system
        system_position = (
            random.randint(0, 1000),
            random.randint(0, 1000),
            random.randint(0, 1000)
        )
        system = StarSystem(
            name=f"System_{sim_id}",
            position=system_position
        )
        galaxy.add_system(system)
        
        # Create test planets with proper spacing
        num_planets = random.randint(3, 7)
        create_planets_with_spacing(system, num_planets, sim_id, seed)
        
        # Get list of habitable planets
        habitable_planets = [p for p in system.planets if p.is_habitable()]
        if not habitable_planets:
            habitable_planets = [system.planets[0]]  # Fallback to first planet
            
        # Create agents and distribute them across habitable planets
        agents = []
        for i in range(10):  # Create 10 agents
            planet = random.choice(habitable_planets)
            agent = Agent(f"Agent_{i}", planet)
            agent.behavior = MLBehavior()  # Add ML-based behavior
            agents.append(agent)
            planet.add_agent(agent)
        
        # Initialize logger
        json_logger = SimulationLogger(sim_id, seed, galaxy)
        
        # Run simulation ticks
        for tick in range(self.ticks_per_simulation):
            # Terminal logging - basic stats
            if self.debug:
                # Change all self.logger.info to print for debug mode
                print(f"\nTick {tick}:")
                alive_agents = sum(1 for agent in agents if agent.alive)
                print(f"Alive agents: {alive_agents}")
                
                # Log planet state using first planet for demonstration
                if system.planets:
                    first_planet = system.planets[0]
                    print("Planet state:")
                    print(f"  Temperature: {first_planet.temperature:.1f}")
                    print(f"  Weather: {first_planet.weather}")
                    print(f"  Hazards: {first_planet.hazards}")
                    print(f"  Resources: {first_planet.resources}")
            
            # Update each agent
            for agent in agents:
                if agent.alive:
                    # Terminal logging - agent state before action
                    if self.debug:
                        print(f"\nAgent {agent.name} state:")
                        print(f"  Energy: {agent.energy:.1f}")
                        print(f"  Health: {agent.health:.1f}")
                        print(f"  Inventory: {agent.inventory}")
                    
                    # Get and execute action
                    action = agent.behavior.decide(agent)
                    success = agent.execute(action)
                    
                    # Track action for statistics
                    if action and success:
                        self.action_counts[action.name] += 1
                    
                    # Terminal logging - action result
                    if self.debug:
                        if action:  # Check if action exists before accessing attributes
                            print(f"  Action: {action.name}")
                            print(f"  Success: {success}")
                            print(f"  Decision reason: {agent.behavior.get_decision_reason()}")
                            print(f"  New energy: {agent.energy:.1f}")
                            print(f"  New health: {agent.health:.1f}")
                            print(f"  New inventory: {agent.inventory}")
                    
                    # Apply environmental effects
                    agent._apply_environmental_effects()
                    
                    # Terminal logging - environmental effects
                    if self.debug:
                        print(f"  After environment:")
                        print(f"    Energy: {agent.energy:.1f}")
                        print(f"    Health: {agent.health:.1f}")
            
            # Update planets
            for planet in system.planets:
                planet.tick()
            
            # JSON logging - comprehensive state
            for planet in system.planets:
                if planet.agents:
                    json_logger.log_tick(planet, planet.agents)
            
            # Check if all agents are dead
            if all(not agent.alive for agent in agents):
                print(f"All agents died at tick {tick}")
                break
        
        print(f"Simulation {sim_id} completed")
        return json_logger

if __name__ == "__main__":
    # Uncomment to test planet spacing with more planets
    # test_planet_spacing()
    # exit()
    
    runner = SimulationRunner(debug=False, ticks_per_simulation=100)
    runner.simulate(ticks=200, output_interval=20, visualize=True)

def test_planet_spacing():
    """Test function to check planet spacing with a higher number of planets."""
    print("Testing planet spacing with a higher number of planets...")
    seed = 12345
    sim_id = 999
    
    # Create galaxy and system structure for testing
    galaxy = Galaxy(name=f"Galaxy_{sim_id}", seed=seed)
    system = StarSystem(
        name=f"System_{sim_id}",
        position=(500, 500, 500)
    )
    galaxy.add_system(system)
    
    # Create test planets with proper spacing - using a higher number to test spacing
    num_planets = 12  # Increased for testing
    create_planets_with_spacing(system, num_planets, sim_id, seed)
    
    print(f"Created {len(system.planets)} planets")
    
    # Print planet positions to verify spacing
    print("\nSystem planets:")
    for i, planet in enumerate(system.planets):
        if hasattr(planet, 'system') and planet.system.name == system.name:
            print(f"Planet {i}: {planet.name} at {planet.coords}")
    
    # Verify minimum distance
    print("\nVerifying minimum distance between planets...")
    min_distance = float('inf')
    closest_pair = None
    
    for i, planet1 in enumerate(system.planets):
        for j, planet2 in enumerate(system.planets):
            if i >= j:  # Skip same planet and already checked pairs
                continue
            
            dist = get_distance(planet1.coords, planet2.coords)
            if dist < min_distance:
                min_distance = dist
                closest_pair = (i, j)
                print(f"New closest: planets {i} and {j}, distance: {dist}")
    
    print(f"Closest planets: {closest_pair}, distance: {min_distance}")
    
    # Fix any planets that are too close to each other
    MIN_REQUIRED_DISTANCE = 200
    if min_distance < MIN_REQUIRED_DISTANCE:
        print(f"\nDetected planets that are too close ({min_distance} < {MIN_REQUIRED_DISTANCE})")
        print("Adjusting planet positions...")
        
        # Move planets away from each other
        for i, planet1 in enumerate(system.planets):
            for j, planet2 in enumerate(system.planets):
                if i >= j:  # Skip same planet and already checked pairs
                    continue
                
                p1_pos = planet1.coords
                p2_pos = planet2.coords
                dist = get_distance(p1_pos, p2_pos)
                
                if dist < MIN_REQUIRED_DISTANCE:
                    print(f"Fixing planets {i} ({planet1.name}) and {j} ({planet2.name})")
                    
                    # Calculate direction from p1 to p2
                    direction_x = p2_pos[0] - p1_pos[0]
                    direction_y = p2_pos[1] - p1_pos[1]
                    direction_z = p2_pos[2] - p1_pos[2]
                    
                    # Normalize direction
                    direction_length = math.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
                    if direction_length > 0:
                        direction_x /= direction_length
                        direction_y /= direction_length
                        direction_z /= direction_length
                    
                    # Move both planets away from each other
                    move_distance = (MIN_REQUIRED_DISTANCE - dist) / 2 + 50  # Extra padding
                    
                    # Move planet1 away from planet2
                    new_p1_x = p1_pos[0] - direction_x * move_distance
                    new_p1_y = p1_pos[1] - direction_y * move_distance
                    new_p1_z = p1_pos[2] - direction_z * move_distance
                    
                    # Move planet2 away from planet1
                    new_p2_x = p2_pos[0] + direction_x * move_distance
                    new_p2_y = p2_pos[1] + direction_y * move_distance
                    new_p2_z = p2_pos[2] + direction_z * move_distance
                    
                    # Update positions
                    planet1.coords = (new_p1_x, new_p1_y, new_p1_z)
                    planet2.coords = (new_p2_x, new_p2_y, new_p2_z)
                    
                    print(f"  New positions:")
                    print(f"  {planet1.name}: {planet1.coords}")
                    print(f"  {planet2.name}: {planet2.coords}")
        
        # Re-check minimum distance
        print("\nVerifying minimum distance after fixes...")
        min_distance = float('inf')
        closest_pair = None
        
        for i, planet1 in enumerate(system.planets):
            for j, planet2 in enumerate(system.planets):
                if i >= j:  # Skip same planet and already checked pairs
                    continue
                
                dist = get_distance(planet1.coords, planet2.coords)
                if dist < min_distance:
                    min_distance = dist
                    closest_pair = (i, j)
        
        print(f"Closest planets after fixes: {closest_pair}, distance: {min_distance}")
    
    # Initialize visualization
    if VISUALIZATION_AVAILABLE:
        print("\nStarting visualization to check spacing visually...")
        visualization_adapter = visualize_simulation(galaxy)
        
        # Run visualization loop
        running = True
        while running:
            if not visualization_adapter.render_and_process_events():
                running = False
            
            time.sleep(0.05)
