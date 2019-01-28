import argparse

# -------------- ARGPARSE PARAMETERS -------------------------------------------
parser = argparse.ArgumentParser(description="Run process data for optical flow")

parser.add_argument('-input',
                    action='store',
                    required=True,
                    type=str,
                    help='''Input event stream (.txt file).
                    Should be a text with a .txt extension
                    Example: inputs/my_input.txt''',
                    metavar='REQUIRED str input_text_file')

parser.add_argument('-output_speed',
                    action='store',
                    required=True,
                    type=str,
                    help='''Output event stream (.txt file).
                    Should be a text with a .txt extension''',
                    metavar='REQUIRED str output_text_file')

parser.add_argument('-width',
                    action='store',
                    required=False,
                    default=240,
                    type=int,
                    help='''Original video intput frame width (int pixel).''',
                    metavar='int frame_width')

parser.add_argument('-height',
                    action='store',
                    required=False,
                    default=304,
                    type=int,
                    help='''Original video intput frame height (int pixel).''',
                    metavar='int frame_width')
