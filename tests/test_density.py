import pytest
from pathlib import Path
import numpy as np
from PIL import Image

from iafoule.density import image_with_density_map

@pytest.fixture
def input_params():
    image_path = Path(__file__).resolve().parent / 'data/image.jpg'
    image = Image.open(image_path)  
    density_map = np.load('data/density_map.npy')
    return image, density_map
   
    
#@pytest.mark.skip(reason="wait later")
@pytest.mark.parametrize("params", [
    ('neg', 0.3, 1.0, 79070290, 30095118),
    ('mean', 0.8, 0.9, 80902903, 29048055),
    ('constant', 0.8, 1.0, 37101870, 65926155),
    ('max', 0.8, 1.0, 82261998, 26006695),
    ('std', 0.5, 1.0, 82176988, 26369684),
])
def test_image_with_density_map_ok(input_params, params):

    image = input_params[0]
    density_map = input_params[1]
    alpha_type = params[0]
    alpha_weight = params[1]
    factor = params[2]

    img_with_dm, img_dm = image_with_density_map(image, density_map, alpha_type=alpha_type, alpha_weight=alpha_weight, factor=factor)
               
    assert isinstance(img_with_dm, Image.Image)
    assert isinstance(img_dm, Image.Image)
    img_with_dm_sum = int(np.array(img_with_dm).sum())
    img_dm_sum = int(np.array(img_dm).sum())
    assert img_with_dm_sum == params[3]
    assert img_dm_sum == params[4]

#@pytest.mark.skip(reason="wait later")
@pytest.mark.parametrize("params", [
    ('xxx', 0.5, 1.0, 82176988, 26369684),
    ('std', -1, 1.0, 82176988, 26369684),
    ('std', 0.5, -1, 82176988, 26369684),
])
def test_image_with_density_map_default(input_params, params):

    image = input_params[0]
    density_map = input_params[1]
    alpha_type = params[0]
    alpha_weight = params[1]
    factor = params[2]

    img_with_dm, img_dm = image_with_density_map(image, density_map, alpha_type=alpha_type, alpha_weight=alpha_weight, factor=factor)
               
    assert isinstance(img_with_dm, Image.Image)
    assert isinstance(img_dm, Image.Image)
    img_with_dm_sum = int(np.array(img_with_dm).sum())
    img_dm_sum = int(np.array(img_dm).sum())
    assert img_with_dm_sum == params[3]
    assert img_dm_sum == params[4]


@pytest.mark.parametrize("params", [
    ('accept', 'accept', 'std', 0.5, 1.0, Image.Image, Image.Image),#valid baseline
    (None, 'accept', 'std', 0.5, 1.0, None, None),#not an PIL image
    ('accept', None, 'std', 0.5, 1.0, None, None),#not an nd array density map
])
def test_image_with_density_map_ko(input_params, params):

    image = input_params[0]
    if params[0]!='accept':
        image = params[0]
    density_map = input_params[1]
    if params[1]!='accept':
        density_map = params[1]
    alpha_type = params[2]
    alpha_weight = params[3]
    factor = params[4]

    img_with_dm, img_dm = image_with_density_map(image, density_map, alpha_type=alpha_type, alpha_weight=alpha_weight, factor=factor)
               
    if params[0]=='accept' and params[1]=='accept':
        assert isinstance(img_with_dm, params[5])
        assert isinstance(img_dm, params[6])
    else:
        assert img_with_dm is None
        assert img_dm is None
    