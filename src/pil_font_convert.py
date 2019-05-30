# Converts a bdf or pcf font file to a
# .pil file to be used with PIL

from PIL import PcfFontFile
f = open("fonts/clean_bold.pcf", 'rb')
PcfFontFile.PcfFontFile(f).save("fonts/clean_bold")
