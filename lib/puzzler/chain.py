import base64
import struct
import zlib

class ChainCode:

    def encode(self, contour):

        code = {(1,0): 0, (1,1): 1, (0,1): 2, (-1,1): 3,
                (-1,0): 4, (-1,-1): 5, (0,-1): 6, (1,-1): 7}

        buffer = bytearray(4 + len(contour)-1)

        x0, y0 = contour[0]
        assert -32768 <= x0 < 32768 and -32768 <= y0 < 32768
        struct.pack_into("!hh", buffer, 0, x0, y0)

        prev = contour[0]
        for i, xy in enumerate(contour[1:], start=4):
            dxdy = (xy[0]-prev[0], xy[1]-prev[1])
            buffer[i] = code[dxdy]
            prev = xy

        assert i+1 == len(buffer)

        cmp_data = zlib.compress(buffer)
        b64_data = base64.standard_b64encode(cmp_data)

        return str(b64_data, encoding='utf-8')

    def decode(self, chain):

        code = {0: (1,0), 1: (1,1), 2: (0,1), 3: (-1,1),
                4: (-1, 0), 5: (-1,-1), 6: (0,-1), 7: (1,-1)}

        cmp_data = base64.standard_b64decode(chain)
        buffer = zlib.decompress(cmp_data)

        x, y = struct.unpack_from("!hh", buffer, 0)
        
        path = [(x,y)]
        for i in buffer[4:]:
            dx, dy = code[i]
            x += dx
            y += dy
            path.append((x, y))

        return path

