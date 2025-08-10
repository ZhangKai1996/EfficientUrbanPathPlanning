import cv2
import numpy as np

from common.utils import haversine_coord

text_kwargs = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=1.5,
                   color=(255, 255, 255),
                   thickness=4)


def latlon_to_pixel(lat, lon, bounder, size):
    min_lat, max_lat, min_lon, max_lon = bounder
    x = int((lon - min_lon) / (max_lon - min_lon) * size[0])
    y = int((lat - min_lat) / (max_lat - min_lat) * size[1])
    return x, size[1] - y


def make_random_color():
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return b, g, r


class StaticRender:
    def __init__(self, graph, height=1440, padding=0.05):
        self.graph = graph

        lats = [graph.nodes[node]['y'] for node in graph.nodes]
        lons = [graph.nodes[node]['x'] for node in graph.nodes]
        # min_lat, max_lat, min_lon, max_lon
        self.bounder = (min(lats), max(lats), min(lons), max(lons))

        # 计算地图长宽比
        delta_lat = (self.bounder[3] - self.bounder[2])
        delta_lon = (self.bounder[1] - self.bounder[0])
        width = int(height * delta_lat / delta_lon)
        self.size = (width, height)
        length = haversine_coord(p1=(self.bounder[0], self.bounder[2]),
                                 p2=(self.bounder[1], self.bounder[2]),
                                 km=True)
        self.density = height / length
        self.padding = (int(width * padding), int(height * padding))

    def __draw_nodes(self, img, vf):
        nodes = self.graph.nodes
        for i, u in enumerate(nodes):
            pt = latlon_to_pixel(nodes[u]['y'], nodes[u]['x'], self.bounder, self.size)
            color = (0, 255, 0)
            if vf is not None and u in vf:
                color = (255, 0, 255)
            cv2.circle(img, pt, 2, color, -1)

        if vf is not None:
            text = 'Nodes: {}'.format(len(vf))
            cv2.putText(img, text, org=(60, 60), **text_kwargs)

    def __draw_edges(self, img, vf):
        nodes = self.graph.nodes
        count = 0
        for u, v, data in self.graph.edges(data=True):
            pt1 = latlon_to_pixel(nodes[u]['y'], nodes[u]['x'], self.bounder, self.size)
            pt2 = latlon_to_pixel(nodes[v]['y'], nodes[v]['x'], self.bounder, self.size)
            color = (0, 255, 0)
            if vf is not None:
                if u in vf and v in vf:
                    color = (255, 0, 255)
                    count += 1
            cv2.line(img, pt1, pt2, color, 1)

        if vf is not None:
            text = 'Edges: {}'.format(count)
            cv2.putText(img, text, org=(60, 120), **text_kwargs)

    def __draw_markers(self, img, node, text):
        nodes = self.graph.nodes
        pt = latlon_to_pixel(nodes[node]['y'], nodes[node]['x'], self.bounder, self.size)
        color = (128, 0, 128)
        cv2.circle(img, pt, 10, color, -1)
        cv2.putText(img, text, org=(pt[0], pt[1] + 60), **text_kwargs)

    def __draw_paths(self, img, paths):
        if paths is None: return

        nodes = self.graph.nodes
        count = 0
        for key, [end_node, cost, path] in paths.items():
            color = make_random_color()
            for i, u in enumerate(path[:-1]):
                v = path[i + 1]
                pt1 = latlon_to_pixel(nodes[u]['y'], nodes[u]['x'], self.bounder, self.size)
                pt2 = latlon_to_pixel(nodes[v]['y'], nodes[v]['x'], self.bounder, self.size)
                cv2.line(img, pt1, pt2, color, 4)
            self.__draw_markers(img, path[0], text='Start')
            self.__draw_markers(img, end_node, text='End')
            text = 'Cost: {} ({})'.format(round(cost, 2), key)
            y = 960 + 60 * count
            cv2.line(img, (10, y-10), (50, y-10), color, 4)
            cv2.putText(img, text, (60, y), **text_kwargs)
            count += 1

    def __draw_range(self, img, ranges):
        if ranges is None: return

        nodes = self.graph.nodes
        center = ranges['center']
        pt = latlon_to_pixel(nodes[center]['y'], nodes[center]['x'], self.bounder, self.size)
        cv2.circle(img, pt, 5, (0, 0, 255), -1)

        radius = int(ranges['radius'] * self.density)
        cv2.circle(img, pt, radius, (255, 255, 255), 1)

        text = 'Center Node: {}'.format(center)
        cv2.putText(img, text, org=(60, 180), **text_kwargs)
        text = 'Radius: {} km'.format(ranges['radius'] / 1e3)
        cv2.putText(img, text, org=(60, 240), **text_kwargs)

    def draw(self, paths=None, vf=None, name=None, ranges=None, **kwargs):
        img = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8) * 255

        self.__draw_nodes(img, vf)
        self.__draw_edges(img, vf)
        self.__draw_paths(img, paths)
        self.__draw_range(img, ranges)

        img_lr = np.zeros((self.size[1], self.padding[0], 3), dtype=np.uint8) * 255
        img_ub = np.zeros((self.padding[1], self.size[0] + self.padding[0] * 2, 3), dtype=np.uint8) * 255
        img = np.hstack([img_lr, img, img_lr])
        img = np.vstack([img_ub, img, img_ub])

        cv2.imshow("City Instant Delivery Simulation", img)
        cv2.waitKey(0)
        if name is not None: cv2.imwrite(name + '.png', img)
        cv2.destroyAllWindows()
