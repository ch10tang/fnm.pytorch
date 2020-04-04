
def args_warning(args):
    print("Parameters:")
    for attr, value in sorted(args.__dict__.items()):
        text = "\t{}={}\n".format(attr.upper(), value)
        print(text)
        with open('{}/Parameters.txt'.format(args.snapshot_dir), 'a') as f:
            f.write(text)

    if args.front_list is None or args.profile_list is None:
        print("Sorry, please set csv-file for your front(normal) /profile(source) data")
        exit()

    if args.data_place is None:
        print("Sorry, please set -data-place for your input data")
        exit()

def evl_args_warning(args):
    if args.encoder is False and args.decoder is False:
        print("Sorry, please set encoder or decoder to true while trigger generate feature")
        exit()
    if args.data_place is None:
        print("Sorry, please set data place for your input dat")
        exit()
    if args.snapshot is None:
        print("Sorry, please set snapshot path while generate")
        exit()