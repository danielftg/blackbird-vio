$out_dir = 'build';
$pdf_mode = 1;
$bibtex_use = 2;

# VS Code is a snap; its child processes inherit broken snap library paths that
# crash inkscape (libpthread from core20). Prepend a local wrapper that strips
# the snap env so the svg package's inkscape calls work correctly.
use Cwd 'abs_path';
use File::Basename;
my $paper_dir = abs_path(dirname(__FILE__));
$ENV{'PATH'} = $paper_dir . ':' . $ENV{'PATH'};

# Export the built PDF to a stable tracked path (build/ itself is gitignored).
END { system('cp', "$paper_dir/build/main.pdf", "$paper_dir/main.pdf") if -f "$paper_dir/build/main.pdf"; }
