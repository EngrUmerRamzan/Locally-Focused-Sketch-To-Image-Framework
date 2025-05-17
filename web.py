import os, shutil
from flask import Flask, render_template, request, redirect, url_for

import torch
from torchvision import transforms
import models, datasets, utils
from basicsr.utils import imwrite
from GFPGAN.gfpgan import GFPGANer
import tempfile

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='CrimSniffer: Inference')
    parser.add_argument('--crimsniffer_weight', type=str, required=True, help='Path to load CrimSniffer model weights.')
    parser.add_argument('--gfpgan_weight', type=str, required=True, help='Path to load GFPGAN model weights.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--manifold', action='store_true', help='Use manifold projection in the model.')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument('--weight', type=float, default=0.5, help='Adjustable weights.')
    args = parser.parse_args()
    return args

class Storage:
    
    def generate_folder(self):
        self.folder_name = str(hash(self))
        while os.path.exists(self.folder_name):
            self.folder_name = str(hash(self.folder_name))
        os.makedirs(self.folder_name)
        os.makedirs(os.path.join(self.folder_name, 'sketch'))
        os.makedirs(os.path.join(self.folder_name, 'photo'))
    
    def delete_folder(self):
        shutil.rmtree(self.folder_name)
    
    def get_folder_path(self):
        return os.path.abspath(self.folder_name)
    
    def __enter__(self):
        self.generate_folder()
        return self
    
    def __exit__(self, *args, **kwargs):
        self.delete_folder()

def main(args, storage):
    
    device = torch.device(args.device)
    print(f'Device : {device}')
    
    crimsniifer_model = models.CrimSniffer(
        CE=True, CE_encoder=True, CE_decoder=False,
        FM=True, FM_decoder=True,
        IS=True, IS_generator=True, IS_discriminator=False,
        manifold=args.manifold
    )
    crimsniifer_model.load(args.crimsniffer_weight)
    crimsniifer_model.to(device)
    crimsniifer_model.eval()

    # Load GFPGAN model
    gfpgan_model = GFPGANer(
        model_path=args.gfpgan_weight,
        upscale=args.upscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None)

    
    template_folder = os.path.abspath('resources/templates/')
    app = Flask(__name__, template_folder=template_folder, static_folder=storage.get_folder_path())
    
    
    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'GET':
            return render_template('index.html')
        if request.method == 'POST':
            sketch = request.files['image']
            file_name = str(hash(str(sketch))) + '.jpg'
            sketch.save(os.path.join(storage.get_folder_path(), 'sketch', file_name))
            return redirect(url_for('forward', file_name=file_name))
    
    @app.route('/forward/<file_name>', methods=['GET'])
    def forward(file_name):
        x = datasets.dataloader.load_one_sketch(os.path.join(storage.get_folder_path(), 'sketch', file_name), simplify=True, device=args.device).unsqueeze(0).to(device)
        x = crimsniifer_model(x)
        x = utils.convert.tensor2PIL(x[0])
        # x.save(os.path.join(storage.get_folder_path(), 'photo', file_name))
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            x.save(temp_file.name)
            temp_file_path = temp_file.name
        cropped_faces, Enhanced_faces, Generated_imgs = gfpgan_model.enhance(
        temp_file_path, # Assuming the enhance method can take a path directly
        has_aligned=args.aligned,
        only_center_face=args.only_center_face,
        paste_back=True,
        weight=args.weight) 

        # Enhanced_faces.save(os.path.join(storage.get_folder_path(), 'photo', file_name))
        for idx, Enhanced_faces in enumerate(Enhanced_faces):
            save_path = os.path.join(storage.get_folder_path(), 'photo', file_name) 
            imwrite(Enhanced_faces, save_path)
        return redirect(url_for('display', file_name=file_name))
    
    @app.route('/display/<file_name>', methods=['GET'])
    def display(file_name):
        return render_template('display.html', file_name=file_name)
        
    host = args.host
    port = args.port
    app.run(host, port)
    
if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    with Storage() as storage:
        main(args, storage)