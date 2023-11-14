# https://developers.google.com/maps/documentation/javascript/coordinates
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates#maps_map_coordinates-javascript
# https://developers.google.com/maps/documentation/tile/2d-tiles-overview
# https://youtu.be/Jz_s21I2M_E?si=hfBrwRXJT2YR9uXP
import requests, math
api_key = ''
token = 'AJVsH2zD-FLrcw0KDSq37g5z59TB9ATlQyRHyfIwQ4BthH8YIrKr_prYFilgiFJaKJFRLHtS0zvgve_1HXbrO_XXZA'
TILE_SIZE = 256
zoom = 16
scale = 1 << zoom


def latlong_to_world(lat, long):
    siny = math.sin((lat * math.pi) / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    return (
        TILE_SIZE * (0.5 + long / 360),
        TILE_SIZE * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))
    )


def world_to_pixel(worldCor):
    px = math.floor(worldCor[0] * scale)
    py = math.floor(worldCor[1] * scale)
    return px, py


def world_to_tile(worldCor):
    tx = math.floor((worldCor[0] * scale) / TILE_SIZE)
    ty = math.floor((worldCor[1] * scale) / TILE_SIZE)
    return tx, ty


# session token is good for 2 weeks
def get_session_token():
    url = 'https://tile.googleapis.com/v1/createSession?key=' + api_key
    headers = {'Content-Type': 'application/json'}
    json_data = {
        'mapType': 'satellite',
        'language': 'en-US',
        'region': 'US',
    }
    response = requests.post(url, headers=headers, json=json_data)
    token = response.json()['session']
    return token


def get_api(file):
    f = open(file, "r")
    line = f.readline()
    return line


def get_tiles(zoom, x, y, direction):
    url = 'https://tile.googleapis.com/v1/2dtiles/' + str(zoom) + '/' + str(x) + '/' + str(y) + \
    '?session=' + token + '&key=' + api_key # + '&orientation=' + str(direction)
    response = requests.get(url)
    print(url)
    print(response)
    return response


if __name__ == '__main__':
    api_key = get_api("key.txt")
    wx, wy = latlong_to_world(40.059698, -88.551633)
    px, py = world_to_pixel((wx, wy))
    tx, ty = world_to_tile((wx, wy))
    get_tiles(zoom, tx, ty, 180)