import os

import svglue
import cairosvg

from research.route_diversity.diversity_settings import CITIES

map_fname = os.path.join("/home/clepe/scratch/diversity_data/maps", "{feed_name}_routes{format}")
distribution_fname = os.path.join("/home/clepe/scratch/diversity_data/maps/distributions",
                                  "{feed_name}_frequency_by_route_type.svg")
legend_fname = os.path.join("/home/clepe/scratch/diversity_data/maps/distributions",
                                  "legend.svg")

for city, feeds in CITIES.items():
    for feed in feeds:
        # load the template from a file
        tpl = svglue.load(file='/home/clepe/scratch/diversity_data/svg_templates/drawing.svg')

        # replace some text
        #tpl.set_text('sample-text', u'This was replaced.')

        # replace the pink box with 'hello.png'. if you do not specify the mimetype,
        # the image will get linked instead of embedded
        tpl.set_svg('map', file=map_fname.format(feed_name=feed, format=".svg"))
        tpl.set_svg('distribution', file=distribution_fname.format(feed_name=feed))

        # svgs are merged into the svg document (i.e. always embedded)
        #tpl.set_svg('yellow-box', file='Ghostscript_Tiger.svg')

        # to render the template, cast it to a string. this also allows passing it
        # as a parameter to set_svg() of another template
        src = str(tpl)

        # write out the result as an SVG image and render it to pdf using cairosvg
        with open('output.pdf', 'w') as out, open('output.svg', 'w') as svgout:
            #svgout.write(src)
            cairosvg.svg2pdf(bytestring=src, write_to=map_fname.format(feed_name=feed, format=".pdf"))
"""
#!/usr/bin/env python



# load the template from a file
tpl = svglue.load(file='/home/clepe/scratch/diversity_data/svg_templates/sample-tpl.svg')

# replace some text
tpl.set_text('sample-text', u'This was replaced.')

# replace the pink box with 'hello.png'. if you do not specify the mimetype,
# the image will get linked instead of embedded
tpl.set_image('pink-box', file='/home/clepe/scratch/diversity_data/svg_templates/hello.png', mimetype='image/png')

# svgs are merged into the svg document (i.e. always embedded)
tpl.set_svg('yellow-box', file='/home/clepe/scratch/diversity_data/svg_templates/Ghostscript_Tiger.svg')

# to render the template, cast it to a string. this also allows passing it
# as a parameter to set_svg() of another template
src = str(tpl)

# write out the result as an SVG image and render it to pdf using cairosvg
with open('/home/clepe/scratch/diversity_data/svg_templates/test.pdf', 'w') as out, open('/home/clepe/scratch/diversity_data/svg_templates/test.svg', 'w') as svgout:
    svgout.write(src)
    cairosvg.svg2pdf(bytestring=src, write_to=out)
    
"""
