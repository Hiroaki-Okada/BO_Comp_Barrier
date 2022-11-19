import pdb


def check_status(instance):
    if instance.second_opt == False and instance.batch_magnification != 1:
        raise ValueError('Invalid batch magnification size')

    if instance.init_method == 'read' and len(instance.pre_real_results) == 0:
        raise ValueError('Invalid known data size')

    if instance.init_method != 'read' and len(instance.pre_real_results) > 0:
        raise ValueError('Invalid known data size')

    if instance.mode.lower() not in ['experiment', 'calculation']:
        raise ValueError('Invalid mode')

    if instance.mode.lower() == 'experiment' and instance.one_step == False:
        raise ValueError('Invalid mode')
