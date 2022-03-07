import sys
incluia_etl = '../../../incluia-etl/'  # Points to the folder that contains 'incluiasrc' folder
sys.path.append(incluia_etl)
from incluiasrc.utils.request_functions import *
from jal.utils.config import *

# lat, lon = 20.648631, -103.367742
# lat1, lat2, lon1, lon2 = corners_coordinates(lat, lon)

ximg_len = 0.0005359965   # np.abs(lon2 - lon1)  # length in long coordinates of a single image.
yimg_len = 0.0004729916   # np.abs(lat2 - lat1) * 375/400 # length in lat coordinates of a single image.
