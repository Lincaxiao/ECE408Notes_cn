# Run each source from its own directory when latexmk is launched from Lecture.
$do_cd = 1;

# The lecture notes use fontspec/xeCJK/unicode-math and therefore require
# XeLaTeX.  Keep command-line latexmk builds consistent with the GUI recipe.
$pdf_mode = 5;

# Put the final PDF next to the source while keeping auxiliary files separate.
$out_dir = '.';
$aux_dir = 'TexOutput';

$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$lualatex = 'lualatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$xelatex = 'xelatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';

$clean_ext .= ' synctex.gz run.xml bcf';
