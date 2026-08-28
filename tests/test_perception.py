import io, sys, unittest
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'agent'))
from perception import TargetTracker, compare_frames

def image(box=None):
    im=Image.new('RGB',(240,160),'black')
    if box:ImageDraw.Draw(im).rectangle(box,fill='white')
    out=io.BytesIO();im.save(out,format='JPEG',quality=95);return out.getvalue()

class PerceptionTests(unittest.TestCase):
    def test_stable_id_and_aim_error(self):
        tracker=TargetTracker(grid=12,threshold=5,max_distance=50);tracker.observe(image())
        first=tracker.observe(image((100,60,119,79)));second=tracker.observe(image((105,60,124,79)))
        self.assertTrue(first['candidates']);self.assertTrue(second['target'])
        self.assertEqual(first['candidates'][0]['id'],second['target']['id']);self.assertIn('distance',second['aim_error'])
    def test_frame_comparison(self):
        self.assertFalse(compare_frames(image(),image())['changed']);self.assertTrue(compare_frames(image(),image((20,20,220,140)))['changed'])
if __name__=='__main__':unittest.main()
