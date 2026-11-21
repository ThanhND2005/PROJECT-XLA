import ij.IJ;
import ij.ImageJ;

public final class GlcmMacroRunner {
    public static void main(String[] args) {
        try {
            // The plugin needs access to ImageJ's Results window through AWT.
            new ImageJ(ImageJ.NO_SHOW);
            String result = IJ.runMacroFile(args[0], args[1]);
            System.exit("[aborted]".equals(result) ? 1 : 0);
        } catch (Exception error) {
            error.printStackTrace();
            System.exit(1);
        }
    }
}
