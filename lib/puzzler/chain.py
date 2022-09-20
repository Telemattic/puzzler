import base64
import zlib

class ChainCode:

    def encode(self, contour):

        code = {(1,0): '0', (1,1): '1', (0,1): '2', (-1,1): '3',
                (-1,0): '4', (-1,-1): '5', (0,-1): '6', (1,-1): '7'}

        path = []
        prev = contour[0]
        for xy in contour[1:]:
            dxdy = (xy[0]-prev[0], xy[1]-prev[1])
            path.append(code[dxdy])
            prev = xy

        raw_data = bytes(''.join(path), 'utf-8')
        cmp_data = zlib.compress(raw_data)
        b64_data = base64.standard_b64encode(cmp_data)

        return str(b64_data, encoding='utf-8')

    def decode(self, chain):

        code = {'0': (1,0), '1': (1,1), '2': (0,1), '3': (-1,1),
                '4': (-1, 0), '5': (-1,-1), '6': (0,-1), '7': (1,-1)}

        cmp_data = base64.standard_b64decode(chain)
        raw_data = zlib.decompress(cmp_data)

        x, y = 0, 0
        
        path = [(x,y)]
        for i in str(raw_data, encoding='utf-8'):
            dx, dy = code[i]
            x += dx
            y += dy
            path.append((x, y))

        return path

