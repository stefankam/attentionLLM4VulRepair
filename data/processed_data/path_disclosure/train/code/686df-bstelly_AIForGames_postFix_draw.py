#pylint: disable = E1101
import pygame
from draw_utils import Rectangle
from draw_utils import Circle
from draw_utils import Line
from vector2 import Vector2
from a_star import AStar
from graph import Graph
from node import Node
from visual_node import GraphVisual
import time

def main():
#Vector2(17, 17)      For center of square
    pygame.init()
    screen_width = 1360
    screen_height = 760
    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill((0, 0, 0))
    <fix/>grid = Graph(27, 19)</fix>
    visual_graph = GraphVisual(grid, 40, screen)
    start_square = pygame.rect.Rect(3, 363, 36, 36)
    goal_square = pygame.rect.Rect(1323, 363, 36, 36)
    start_node = Node(Vector2(0, 9))
    goal_node = Node(Vector2(33, 9))
    astar = AStar(grid, start_node, goal_node)
    animate_path = []
    drawn_path = []
    path = []
    iterator = 0
    iterator_two = 1

    dragging_start = False
    dragging_goal = False
    mouse_is_down = False
    pressed_enter = False
    
    while True:
        screen.fill((0, 0, 0))
        visual_graph = GraphVisual(grid, 40, screen)
#        goal_node = Node(Vector2(goal_square.left, goal_square.top))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        pygame.event.pump()
        if event.type == pygame.MOUSEBUTTONDOWN:
            pressed_enter = False
            del path[:]
            if event.button == 1:
                if start_square.collidepoint(event.pos):
                    dragging_start = True
                    mouse_x, mouse_y = event.pos
                    offset_x = start_square.x - mouse_x
                    offset_y = start_square.y - mouse_y
                elif goal_square.collidepoint(event.pos):
                    dragging_goal = True
                    mouse_x, mouse_y = event.pos
                    offset_x = goal_square.x - mouse_x
                    offset_y = goal_square.y - mouse_y
                else:
                    count = 0
                    for node in visual_graph.node_visual_colliders:
                        if node.collidepoint(event.pos) and mouse_is_down is False:
                            current_state = grid.nodes[count].is_traversable
                            grid.nodes[count].toggle_state("wall")
                            mouse_is_down = True
                        count += 1
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_is_down = False
            if event.button == 1:
                count = 0
                if dragging_start is True or dragging_goal is True:
                    for collider in visual_graph.node_visual_colliders:
                        if start_square.colliderect(collider):
                            start_square.left = visual_graph.node_visual_colliders[count].left
                            start_square.top = visual_graph.node_visual_colliders[count].top
                            dragging_start = False
                            start_node = Node(Vector2(visual_graph.node_visuals[count].node.get_x(), visual_graph.node_visuals[count].node.get_y()))
                        if goal_square.colliderect(collider):
                            goal_square.left = visual_graph.node_visual_colliders[count].left
                            goal_square.top = visual_graph.node_visual_colliders[count].top
                            dragging_goal = False
                            goal_node = Node(Vector2(visual_graph.node_visuals[count].node.get_x(), visual_graph.node_visuals[count].node.get_y()))
                        count += 1
                    astar = AStar(grid, start_node, goal_node)
        elif event.type == pygame.MOUSEMOTION:
            if dragging_start:
                mouse_x, mouse_y = event.pos
                start_square.x = mouse_x + offset_x
                start_square.y = mouse_y + offset_y
            elif dragging_goal:
                mouse_x, mouse_y = event.pos
                goal_square.x = mouse_x + offset_x
                goal_square.y = mouse_y + offset_y
            elif mouse_is_down:
                count = 0
                for node in visual_graph.node_visual_colliders:
                    if node.collidepoint(event.pos) and grid.nodes[count].is_traversable is current_state:
                        grid.nodes[count].toggle_state("wall")
                    count += 1
        
        if pygame.key.get_pressed()[pygame.K_c]:
            for x in range(0, grid.length * grid.height):
                if grid.nodes[x].is_traversable is False:
                    grid.nodes[x].toggle_state("wall")
                    path = astar.find_path()
                pressed_enter = False
                
        if pygame.key.get_pressed()[pygame.K_RETURN]:
            pressed_enter = True
            if not path:
                path = astar.find_path()
            else:
                astar = AStar(grid, start_node, goal_node)
                path = astar.find_path()

        if pressed_enter:
            time.sleep(.05)
            count = 0
            count_two = 1
            <fix/>if iterator_two <= len(path) - 1:
                line_start = Vector2(path[iterator].get_x() * 40, path[iterator].get_y() * 40)
                line_end = Vector2(path[iterator_two].get_x() * 40, path[iterator_two].get_y() * 40)
                animate_path.append(Line(screen, (255, 255, 0), Vector2(line_start.x_pos + 20,
                                                                    line_start.y_pos + 20),
                                       Vector2(line_end.x_pos + 20,
                                               line_end.y_pos + 20), 5))
            while count_two <= len(animate_path):</fix>
                line_start = Vector2(path[count].get_x() * 40, path[count].get_y() * 40)
                line_end = Vector2(path[count_two].get_x() * 40, path[count_two].get_y() * 40)
                <fix/>drawn_path.append(Line(screen, (255, 255, 0), Vector2(line_start.x_pos + 20,</fix>
                                       line_start.y_pos + 20), Vector2(line_end.x_pos + 20,
                                       line_end.y_pos + 20), 5))
                count += 1
                count_two += 1
            iterator += 1
            iterator_two += 1
        else:
            iterator = 0
            iterator_two = 1
            del animate_path[:]
            del drawn_path[:]

        count = 0
        for node in grid.nodes:
            if node.is_traversable is False:
                pygame.draw.rect(screen, (0, 0, 0), visual_graph.node_visual_colliders[count])
            count += 1
        <fix/>pygame.draw.rect(screen, (0, 255, 0), start_square)
        pygame.draw.rect(screen, (255, 0, 0), goal_square)</fix>
        pygame.display.flip()

main()
