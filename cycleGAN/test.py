import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html

def norm(tensor):
    r = tensor.max() - tensor.min()
    tensor = (tensor - tensor.min())/r
    return tensor

def mxAxis(tensor):
    _, indices = torch.max(tensor, 0)
    return indices


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 0   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test

    for i, data in enumerate(dataset):

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image %04d' % (i, len(dataset)))
        # import pdb; pdb.set_trace()
        # for i in range(opt.batchSize):
        #     vs2 = {}
        #     for k, v in visuals.items():
        #         vs2[k] = v
        #     save_images(webpage, vs2, img_path[i], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()
