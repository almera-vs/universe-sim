"""
2D visualization of the universe simulation using Pygame.
"""
import pygame
import time
import math
import random
from typing import List, Dict, Any, Tuple, Optional

# Color constants
COLORS = {
    "background": (5, 5, 15),
    "star": (255, 255, 200),
    "planet_outline": (100, 100, 150),
    "agent": (255, 150, 0),
    "agent_migrating": (0, 255, 255),
    "agent_dead": (150, 0, 0),
    "text": (255, 255, 255),
    "text_bg": (10, 10, 30, 180),  # Semi-transparent background for text
    "food": (0, 200, 0),
    "water": (0, 100, 255),
    "minerals": (150, 75, 0),
    "energy": (255, 255, 0),
    "gases": (200, 200, 200),
    "temperate": (0, 255, 100),
    "garden": (50, 200, 50),
    "oceanic": (0, 100, 200),
    "lush": (0, 150, 0),
    "forest": (0, 100, 0),
    "savanna": (150, 150, 0),
    "desert": (200, 200, 100),
    "tundra": (200, 200, 255),
    "volcanic": (200, 0, 0),
    "toxic": (150, 0, 150),
}

class TextPosition:
    """Helper class to track text positions and detect collisions."""
    def __init__(self, screen=None):
        self.positions = []  # List of (x, y, width, height) rectangles
        self.screen = screen
        
    def add_position(self, x, y, width, height):
        """Add a text position."""
        self.positions.append((x, y, width, height))
        
    def check_collision(self, x, y, width, height):
        """Check if a new text position would collide with existing ones."""
        rect1 = pygame.Rect(x, y, width, height)
        
        for pos_x, pos_y, pos_width, pos_height in self.positions:
            rect2 = pygame.Rect(pos_x, pos_y, pos_width, pos_height)
            if rect1.colliderect(rect2):
                return True
                
        return False
        
    def find_free_position(self, x, y, width, height, planet_radius, position_options=None):
        """Find a free position for text near a planet."""
        if position_options is None:
            # More comprehensive list of positions to try in different directions
            position_options = [
                (0, planet_radius + 5),                # Bottom
                (planet_radius + 5, 0),                # Right
                (0, -planet_radius - height - 5),      # Top
                (-width - 5, 0),                       # Left
                (planet_radius + 5, -height - 5),      # Top-right
                (-width - 5, planet_radius + 5),       # Bottom-left
                (-width - 5, -height - 5),             # Top-left
                (planet_radius + 5, planet_radius + 5) # Bottom-right
            ]
            
            # Add diagonal positions
            diag_offset = int(planet_radius * 0.7)
            diag_positions = [
                (diag_offset, diag_offset),                # Bottom-right diagonal
                (-width - diag_offset, diag_offset),       # Bottom-left diagonal
                (diag_offset, -height - diag_offset),      # Top-right diagonal
                (-width - diag_offset, -height - diag_offset)  # Top-left diagonal
            ]
            position_options.extend(diag_positions)
        
        # Screen dimensions for boundary checking
        screen_width = 1200
        screen_height = 800
        if self.screen:
            screen_width = self.screen.get_width()
            screen_height = self.screen.get_height()
        
        # Try each position option with increasing distance
        for multiplier in range(1, 6):  # Try distances from 1x to 5x
            for dx, dy in position_options:
                new_x = x + dx * multiplier
                new_y = y + dy * multiplier
                
                # Ensure text stays within screen bounds
                if new_x < 5:
                    new_x = 5
                if new_y < 5:
                    new_y = 5
                if new_x + width > screen_width - 5:
                    new_x = screen_width - width - 5
                if new_y + height > screen_height - 5:
                    new_y = screen_height - height - 5
                    
                if not self.check_collision(new_x, new_y, width, height):
                    self.add_position(new_x, new_y, width, height)
                    return new_x, new_y
        
        # If still no position found, try with larger offsets and wider spacing
        for extra_distance in range(50, 200, 30):  # Try with extra distance in 30px increments
            for angle in range(0, 360, 30):  # Try 12 different angles
                angle_rad = math.radians(angle)
                dx = math.cos(angle_rad) * (planet_radius + extra_distance)
                dy = math.sin(angle_rad) * (planet_radius + extra_distance)
                
                new_x = x + dx - width/2  # Center text around this point
                new_y = y + dy - height/2
                
                # Ensure text stays within screen bounds
                if new_x < 5:
                    new_x = 5
                if new_y < 5:
                    new_y = 5
                if new_x + width > screen_width - 5:
                    new_x = screen_width - width - 5
                if new_y + height > screen_height - 5:
                    new_y = screen_height - height - 5
                    
                if not self.check_collision(new_x, new_y, width, height):
                    self.add_position(new_x, new_y, width, height)
                    return new_x, new_y
        
        # If we still can't find a position, just place it at the bottom of the screen with minimal overlap
        # Find the least crowded area at the bottom of the screen
        best_position = (x, y)  # Default to original position
        min_collisions = float('inf')
        
        # Try positions along the bottom of the screen
        for test_x in range(10, screen_width - width - 10, 20):
            test_y = screen_height - height - 10
            
            # Count collisions
            collision_count = 0
            test_rect = pygame.Rect(test_x, test_y, width, height)
            for pos_x, pos_y, pos_width, pos_height in self.positions:
                other_rect = pygame.Rect(pos_x, pos_y, pos_width, pos_height)
                if test_rect.colliderect(other_rect):
                    collision_count += 1
            
            if collision_count < min_collisions:
                min_collisions = collision_count
                best_position = (test_x, test_y)
        
        self.add_position(best_position[0], best_position[1], width, height)
        return best_position[0], best_position[1]

class UniverseRenderer:
    """Pygame-based renderer for universe simulation."""
    
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.running = False
        self.paused = False
        self.fps = 60
        
        # Simulation data
        self.systems = []
        self.planets = []
        self.agents = []
        self.current_tick = 0
        
        # Visualization state
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.selected_planet = None
        self.show_details = True
        self.show_resources = True
        
        # Planet visualization
        self.planet_radius_base = 30
        self.agent_radius = 5
        
        # Create fixed star positions in a much larger space (3x screen size)
        # This allows smooth panning without stars jumping
        self.star_space_width = width * 3
        self.star_space_height = height * 3
        self.stars = []
        for _ in range(500):
            x = random.randint(0, self.star_space_width)
            y = random.randint(0, self.star_space_height)
            brightness = random.uniform(0.2, 1.0)
            twinkle_speed = random.uniform(1.0, 3.0)
            self.stars.append((x, y, brightness, twinkle_speed))
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 12)
        self.font_medium = pygame.font.SysFont("Arial", 14)
        self.font_large = pygame.font.SysFont("Arial", 18)
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Universe Simulation Visualizer")
        self.clock = pygame.time.Clock()
        
        # For tracking text positions to avoid overlap
        self.text_positions = None
    
    def start(self):
        """Start the visualization."""
        self.running = True
    
    def stop(self):
        """Stop the visualization."""
        self.running = False
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
    
    def update_data(self, systems=None, planets=None, agents=None, current_tick=None):
        """Update the visualization data."""
        if systems is not None:
            self.systems = systems
        if planets is not None:
            self.planets = planets
        if agents is not None:
            self.agents = agents
        if current_tick is not None:
            self.current_tick = current_tick
    
    def process_events(self):
        """Process Pygame events and return False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.toggle_pause()
                elif event.key == pygame.K_d:
                    self.show_details = not self.show_details
                elif event.key == pygame.K_r:
                    self.show_resources = not self.show_resources
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.zoom *= 1.1
                elif event.key == pygame.K_MINUS:
                    self.zoom /= 1.1
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_click(event.pos)
                elif event.button == 4:  # Scroll up
                    self.zoom *= 1.1
                elif event.button == 5:  # Scroll down
                    self.zoom /= 1.1
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[2]:  # Right mouse button drag
                    self.offset_x += event.rel[0]
                    self.offset_y += event.rel[1]
        return True
    
    def render_frame(self):
        """Render a single frame."""
        if not self.running:
            return
            
        # Clear screen
        self.screen.fill(COLORS["background"])
        
        # Reset text position tracking
        self.text_positions = TextPosition(self.screen)
        
        # Draw stars in background
        self._draw_stars()
        
        # Draw systems
        for system in self.systems:
            self._draw_system(system)
        
        # Draw planets - sort by z-coordinate to handle drawing order
        sorted_planets = sorted(self.planets, key=lambda p: getattr(p, 'coords', (0, 0, 0))[2] if hasattr(p, 'coords') else 0)
        for planet in sorted_planets:
            self._draw_planet(planet)
        
        # Draw details for selected planet
        if self.selected_planet and self.show_details:
            self._draw_planet_details(self.selected_planet)
        
        # Draw UI elements
        self._draw_ui()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _handle_click(self, pos):
        """Handle mouse click on the visualization."""
        clicked_planet = None
        for planet in self.planets:
            x, y = self._get_planet_position(planet)
            radius = self._get_planet_radius(planet)
            
            # Check if click is inside planet
            dist = math.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
            if dist <= radius:
                clicked_planet = planet
                break
        
        self.selected_planet = clicked_planet
    
    def _draw_stars(self):
        """Draw background stars with improved performance."""
        current_time = time.time()
        
        # Calculate visible region of star space based on offset
        # This ensures stars appear fixed in space during panning
        view_left = -self.offset_x % self.star_space_width
        view_top = -self.offset_y % self.star_space_height
        
        for x, y, brightness, twinkle_speed in self.stars:
            # Calculate position with offset wrap-around
            star_x = (x - view_left) % self.star_space_width
            star_y = (y - view_top) % self.star_space_height
            
            # Only draw stars that are within the screen bounds
            if 0 <= star_x < self.width and 0 <= star_y < self.height:
                # Twinkle effect
                brightness_mod = 0.7 + 0.3 * math.sin(current_time * twinkle_speed)
                
                # Calculate color with brightness
                color = (
                    min(255, int(COLORS["star"][0] * brightness * brightness_mod)),
                    min(255, int(COLORS["star"][1] * brightness * brightness_mod)),
                    min(255, int(COLORS["star"][2] * brightness * brightness_mod))
                )
                
                # Draw star based on brightness
                size = 1 if brightness < 0.8 else 2
                pygame.draw.circle(self.screen, color, (int(star_x), int(star_y)), size)
    
    def _draw_system(self, system):
        """Draw a star system."""
        # Get system position
        x, y = self._get_system_position(system)
        
        # Draw star
        pygame.draw.circle(self.screen, COLORS["star"], (x, y), int(15 * self.zoom))
        
        # Draw system name
        if self.show_details:
            name_text = self.font.render(system.name, True, COLORS["text"])
            
            # Find a non-overlapping position for the text
            text_width = name_text.get_width()
            text_height = name_text.get_height()
            
            # Try to find a free position for the text, starting below the star
            text_x, text_y = self.text_positions.find_free_position(
                x - text_width // 2, 
                y + int(20 * self.zoom),
                text_width,
                text_height,
                int(15 * self.zoom)
            )
            
            # Draw text with a background for better visibility
            self._draw_text_with_background(name_text, text_x, text_y)
    
    def _draw_planet(self, planet):
        """Draw a planet with its agents and resources."""
        # Get planet position
        x, y = self._get_planet_position(planet)
        radius = self._get_planet_radius(planet)
        
        # Determine planet color based on type
        planet_color = COLORS.get(planet.planet_type, COLORS["planet_outline"])
        
        # Draw planet
        pygame.draw.circle(self.screen, planet_color, (x, y), radius)
        pygame.draw.circle(self.screen, COLORS["planet_outline"], (x, y), radius, max(1, int(self.zoom)))
        
        # Draw resource indicators if enabled
        if self.show_resources:
            self._draw_resource_indicators(planet, x, y, radius)
        
        # Draw agents on the planet
        self._draw_agents(planet, x, y, radius)
        
        # Draw planet name and agent count with background to improve readability
        name_text = self.font.render(planet.name, True, COLORS["text"])
        
        # Get the text position avoiding overlaps
        text_width = name_text.get_width()
        text_height = name_text.get_height()
        
        # Try different positioning strategies for planet labels
        # Create a list of potential positions to try, ordered by preference
        position_options = [
            # Format: (dx, dy) relative to planet center
            (0, radius + 10),                 # Bottom
            (radius + 10, 0),                 # Right
            (0, -radius - text_height - 10),  # Top
            (-text_width - 10, 0),            # Left
            (radius + 10, -text_height - 10), # Top-right
            (-text_width - 10, radius + 10),  # Bottom-left
            (radius + 10, radius + 10),       # Bottom-right
            (-text_width - 10, -text_height - 10)  # Top-left
        ]
        
        # Find a free position for the planet name text that doesn't overlap
        text_x, text_y = self.text_positions.find_free_position(
            x - text_width // 2, 
            y + radius + 5,
            text_width,
            text_height,
            radius,
            position_options
        )
        
        # Draw text with background
        self._draw_text_with_background(name_text, text_x, text_y)
        
        # Draw agent count below the planet name or in another free position
        if hasattr(planet, 'agents'):
            alive_agents = sum(1 for agent in planet.agents if agent.alive)
            agents_text = self.font.render(f"Agents: {alive_agents}/{len(planet.agents)}", True, COLORS["text"])
            
            # Position agent count text below planet name
            agents_text_width = agents_text.get_width()
            agents_text_height = agents_text.get_height()
            
            # Try to place agent count below the planet name first
            preferred_x = text_x + (text_width - agents_text_width) // 2
            preferred_y = text_y + text_height + 2
            
            # Check if this position is free, otherwise find another free position
            if self.text_positions.check_collision(preferred_x, preferred_y, agents_text_width, agents_text_height):
                # If the preferred position collides, find another free position
                agents_x, agents_y = self.text_positions.find_free_position(
                    preferred_x,
                    preferred_y,
                    agents_text_width,
                    agents_text_height,
                    radius,
                    position_options
                )
            else:
                # Use the preferred position
                agents_x, agents_y = preferred_x, preferred_y
                self.text_positions.add_position(agents_x, agents_y, agents_text_width, agents_text_height)
            
            # Draw agent count text with background
            self._draw_text_with_background(agents_text, agents_x, agents_y)
    
    def _draw_resource_indicators(self, planet, x, y, radius):
        """Draw resource level indicators around the planet."""
        if not hasattr(planet, 'resources'):
            return
            
        # Draw food and water levels as arcs around the planet
        resources = planet.resources
        
        # Food level (green arc)
        if 'food' in resources:
            food_pct = min(1.0, resources['food'] / 1000)
            if food_pct > 0:
                start_angle = 0
                end_angle = food_pct * 2 * math.pi
                rect = pygame.Rect(x - radius - 5, y - radius - 5, 2 * radius + 10, 2 * radius + 10)
                pygame.draw.arc(self.screen, COLORS["food"], rect, start_angle, end_angle, 3)
        
        # Water level (blue arc)
        if 'water' in resources:
            water_pct = min(1.0, resources['water'] / 1000)
            if water_pct > 0:
                start_angle = math.pi
                end_angle = start_angle + water_pct * 2 * math.pi
                rect = pygame.Rect(x - radius - 10, y - radius - 10, 2 * radius + 20, 2 * radius + 20)
                pygame.draw.arc(self.screen, COLORS["water"], rect, start_angle, end_angle, 3)
    
    def _draw_agents(self, planet, planet_x, planet_y, planet_radius):
        """Draw agents on the planet."""
        if not hasattr(planet, 'agents') or not planet.agents:
            return
            
        # Calculate positions in a circle around the planet
        agent_count = len(planet.agents)
        
        # Sort agents by ID to make them appear in a consistent order
        sorted_agents = sorted(planet.agents, key=lambda a: a.id)
        
        for i, agent in enumerate(sorted_agents):
            # Calculate position on the planet's circumference
            angle = (i / agent_count) * 2 * math.pi
            distance = planet_radius * 0.8  # Place slightly inside the planet
            
            agent_x = planet_x + int(math.cos(angle) * distance)
            agent_y = planet_y + int(math.sin(angle) * distance)
            
            # Determine agent color based on state
            if not agent.alive:
                color = COLORS["agent_dead"]
            elif agent.migration_state:
                color = COLORS["agent_migrating"]
            else:
                # Color based on health
                health_ratio = agent.health / 100
                color = (
                    int(255 * (1 - health_ratio)),
                    int(200 * health_ratio),
                    0
                )
            
            # Calculate agent radius based on zoom
            agent_radius = int(self.agent_radius * self.zoom)
            agent_radius = max(3, agent_radius)  # Ensure minimum size
            
            # Draw agent outline for better visibility
            outline_color = (0, 0, 0)
            pygame.draw.circle(
                self.screen, 
                outline_color, 
                (agent_x, agent_y), 
                agent_radius + 1  # Slightly larger for outline
            )
            
            # Draw agent
            pygame.draw.circle(
                self.screen, 
                color, 
                (agent_x, agent_y), 
                agent_radius
            )
            
            
    def _draw_text_with_background(self, text_surface, x, y):
        """Draw text with a semi-transparent background for better readability."""
        padding = 4  # Increased from 2 to 4 for better visibility
        bg_rect = pygame.Rect(x - padding, y - padding, 
                             text_surface.get_width() + padding * 2, 
                             text_surface.get_height() + padding * 2)
        
        # Create a transparent background surface
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        
        # Use a darker, more opaque background for better readability
        bg_color = list(COLORS["text_bg"])
        bg_color[3] = 220  # More opaque (0-255)
        bg_surface.fill(tuple(bg_color))
        
        # Draw background and then text
        self.screen.blit(bg_surface, bg_rect)
        self.screen.blit(text_surface, (x, y))
        
        # Optional: Add a subtle border around the text for even better visibility
        border_color = (30, 30, 60, 180)
        pygame.draw.rect(self.screen, border_color, bg_rect, 1)  # 1 pixel border
    
    def _draw_planet_details(self, planet):
        """Draw detailed information about the selected planet."""
        # Position for the details panel
        panel_x = 10
        panel_y = 10
        panel_width = 300
        panel_height = 300
        
        # Draw semi-transparent panel
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 30, 200))
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Draw planet details
        title_text = self.font_large.render(f"Planet: {planet.name}", True, COLORS["text"])
        self.screen.blit(title_text, (panel_x + 10, panel_y + 10))
        
        y_offset = panel_y + 40
        line_height = 20
        
        # Basic planet info
        details = [
            f"Type: {planet.planet_type}",
            f"Temperature: {planet.temperature:.1f}°C",
            f"Weather: {planet.weather}",
            f"Hazards: {', '.join(planet.hazards) if planet.hazards else 'None'}",
            f"Day/Night: {'Day' if planet.is_day else 'Night'}",
            f"Population: {len(planet.agents)}/{planet.max_population}"
        ]
        
        for detail in details:
            detail_text = self.font_medium.render(detail, True, COLORS["text"])
            self.screen.blit(detail_text, (panel_x + 10, y_offset))
            y_offset += line_height
        
        # Resources
        y_offset += 10
        resource_title = self.font_medium.render("Resources:", True, COLORS["text"])
        self.screen.blit(resource_title, (panel_x + 10, y_offset))
        y_offset += line_height
        
        for resource, amount in planet.resources.items():
            resource_color = COLORS.get(resource, COLORS["text"])
            
            # Draw resource bar
            bar_width = 150
            fill_width = int(bar_width * min(1.0, amount / 1000))
            pygame.draw.rect(self.screen, (50, 50, 50), (panel_x + 110, y_offset + 2, bar_width, 15))
            pygame.draw.rect(self.screen, resource_color, (panel_x + 110, y_offset + 2, fill_width, 15))
            
            resource_text = self.font.render(f"{resource.capitalize()}: {amount:.0f}", True, resource_color)
            self.screen.blit(resource_text, (panel_x + 10, y_offset))
            y_offset += line_height
        
        # Agents summary
        if hasattr(planet, 'agents') and planet.agents:
            y_offset += 10
            alive_agents = sum(1 for agent in planet.agents if agent.alive)
            agents_title = self.font_medium.render(
                f"Agents: {alive_agents} alive / {len(planet.agents)} total", 
                True, COLORS["text"]
            )
            self.screen.blit(agents_title, (panel_x + 10, y_offset))
    
    def _draw_ui(self):
        """Draw UI elements like status and controls."""
        # Draw simulation status
        status_text = "PAUSED" if self.paused else "RUNNING"
        status_color = (255, 100, 100) if self.paused else (100, 255, 100)
        status_render = self.font_large.render(status_text, True, status_color)
        self.screen.blit(status_render, (self.width - status_render.get_width() - 10, 10))
        
        # Draw current tick
        tick_text = f"Tick: {self.current_tick}"
        tick_render = self.font_large.render(tick_text, True, COLORS["text"])
        self.screen.blit(tick_render, (self.width - tick_render.get_width() - 10, 40))
        
        # Draw controls help
        controls = [
            "ESC: Quit",
            "Space: Pause/Resume",
            "D: Toggle Details",
            "R: Toggle Resources",
            "+/-: Zoom In/Out",
            "Right Click + Drag: Pan",
            "Left Click: Select Planet"
        ]
        
        y_pos = self.height - 10 - len(controls) * 15
        for control in controls:
            control_render = self.font.render(control, True, COLORS["text"])
            self.screen.blit(control_render, (10, y_pos))
            y_pos += 15
    
    def _get_system_position(self, system):
        """Calculate the screen position of a system."""
        base_x = self.width // 2
        base_y = self.height // 2
        
        # Get system coordinates
        if hasattr(system, 'position'):
            system_x, system_y, _ = system.position
        else:
            # Use system name to generate a position if coordinates not available
            random.seed(hash(system.name))
            system_x = random.randint(0, 1000) - 500
            system_y = random.randint(0, 1000) - 500
        
        # Apply zoom and offset
        screen_x = int(base_x + system_x * self.zoom * 0.2) + self.offset_x
        screen_y = int(base_y + system_y * self.zoom * 0.2) + self.offset_y
        
        return screen_x, screen_y
    
    def _get_planet_position(self, planet):
        """Calculate the screen position of a planet."""
        # Get system position if planet has a system
        if hasattr(planet, 'system'):
            base_x, base_y = self._get_system_position(planet.system)
        else:
            base_x = self.width // 2
            base_y = self.height // 2
        
        # Get planet orbit information
        if hasattr(planet, 'coords'):
            planet_x, planet_y, _ = planet.coords
        else:
            # Only generate position if coords not available
            # But we shouldn't reach this case for properly initialized planets
            planet_seed = hash(planet.name) if hasattr(planet, 'name') else 0
            random.seed(planet_seed)
            planet_x = random.randint(-400, 400)  # Use a wider range for better spacing
            planet_y = random.randint(-400, 400)
            print(f"Warning: Planet {planet.name} has no coordinates, generating random position")
        
        # Apply zoom and positioning
        screen_x = int(base_x + planet_x * self.zoom * 0.5)
        screen_y = int(base_y + planet_y * self.zoom * 0.5)
        
        return screen_x, screen_y
    
    def _get_planet_radius(self, planet):
        """Calculate the radius to draw the planet based on zoom level and planet properties."""
        base_radius = self.planet_radius_base
        
        # Adjust size based on planet type if available
        if hasattr(planet, 'planet_type'):
            type_size_multiplier = {
                "temperate": 1.2,
                "garden": 1.3,
                "oceanic": 1.4,
                "lush": 1.2,
                "forest": 1.1,
                "savanna": 1.0,
                "desert": 0.9,
                "tundra": 0.8,
                "volcanic": 0.7,
                "toxic": 0.6
            }.get(planet.planet_type, 1.0)
            
            base_radius *= type_size_multiplier
        
        # Apply zoom
        radius = int(base_radius * self.zoom)
        
        # Ensure minimum size
        return max(5, radius) 