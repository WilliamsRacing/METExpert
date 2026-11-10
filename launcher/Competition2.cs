using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

internal static class Competition2 {
    private static int Main(string[] args) {
        Console.OutputEncoding = new UTF8Encoding(false);
        bool noPause = args.Contains("--no-pause");
        bool checkOnly = args.Contains("--check-only");
        if (args.Contains("--help")) {
            Console.WriteLine("Competition2.exe [--check-only] [--no-pause]");
            return 0;
        }
        if (args.Any(a => a != "--no-pause" && a != "--check-only")) {
            Console.Error.WriteLine("Unknown argument. Use --help.");
            return 2;
        }
        int code = 1;
        try {
            string root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
            string script = Path.Combine(root, "launcher", "bootstrap.ps1");
            if (!File.Exists(script)) throw new FileNotFoundException("Keep the complete model folder together.", script);
            var info = new ProcessStartInfo("powershell.exe",
                "-NoLogo -NoProfile -ExecutionPolicy Bypass -File \"" + script + "\" -Root \"" + root + "\"" + (checkOnly ? " -CheckOnly" : ""));
            info.WorkingDirectory = root;
            info.UseShellExecute = false;
            using (var process = Process.Start(info)) { process.WaitForExit(); code = process.ExitCode; }
            Console.WriteLine(code == 0 ? "Completed successfully." : "Run failed. See logs/bootstrap.log.");
        } catch (Exception error) { Console.Error.WriteLine(error.Message); }
        if (!noPause) { Console.WriteLine("Press Enter to close."); Console.ReadLine(); }
        return code;
    }
}
