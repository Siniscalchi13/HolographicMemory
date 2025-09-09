import Cocoa

@main
class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var server: PythonServerController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Create menu bar item
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            button.title = "Holo"
        }
        let menu = NSMenu()
        menu.addItem(NSMenuItem(title: "Open HoloDesk", action: #selector(openWeb), keyEquivalent: "o"))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(quit), keyEquivalent: "q"))
        statusItem.menu = menu

        // Start Python FastAPI server (embedded)
        server = PythonServerController()
        do {
            try server?.start()
            // Open the web interface
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                self.openWeb()
            }
        } catch {
            showErrorAndQuit("Failed to start server: \(error.localizedDescription)")
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        server?.stop()
    }

    @objc private func openWeb() {
        if let url = URL(string: "http://localhost:8000/") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func quit() {
        NSApplication.shared.terminate(nil)
    }

    private func showErrorAndQuit(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "HolographicMemory Error"
        alert.informativeText = message
        alert.alertStyle = .critical
        alert.runModal()
        NSApplication.shared.terminate(nil)
    }
}

final class PythonServerController {
    private var process: Process?

    func start() throws {
        guard process == nil else { return }

        let bundle = Bundle.main
        let fm = FileManager.default

        // Resolve Python executable inside bundled Python.framework
        // Prefer Python.app binary inside the framework's Resources
        let fwBase = (bundle.privateFrameworksPath as NSString?)?.appendingPathComponent("Python.framework")
        let pyApp = fwBase.map { ($0 as NSString).appendingPathComponent("Resources/Python.app/Contents/MacOS/Python") }
        let pyBin = fwBase.map { ($0 as NSString).appendingPathComponent("Versions/3.12/bin/python3") }

        guard let pyExec = [pyApp, pyBin].compactMap({ $0 }).first(where: { fm.isExecutableFile(atPath: $0) }) else {
            throw NSError(domain: "Holo", code: 1, userInfo: [NSLocalizedDescriptionKey: "Embedded Python not found. Run scripts/macos/prepare_python_env.sh"])
        }

        // Compute paths
        let resources = bundle.resourcePath ?? ""
        let pythonHome = fwBase.map { ($0 as NSString).appendingPathComponent("Versions/3.12") } ?? ""
        let sitePackages = (resources as NSString).appendingPathComponent("Python/site-packages")

        // Ensure data dir exists under Application Support
        let appSupport = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dataDir = appSupport.appendingPathComponent("HolographicMemory", isDirectory: true)
        try? FileManager.default.createDirectory(at: dataDir, withIntermediateDirectories: true)

        // Launch uvicorn: services.api.app:app
        let p = Process()
        p.executableURL = URL(fileURLWithPath: pyExec)
        p.arguments = [
            "-m", "uvicorn",
            "services.api.app:app",
            "--host", "127.0.0.1",
            "--port", "8000"
        ]

        var env = ProcessInfo.processInfo.environment
        env["PYTHONHOME"] = pythonHome
        // Ensure embedded framework + OpenSSL are discoverable
        let fwDir = ((bundle.privateFrameworksPath ?? "") as NSString).appendingPathComponent("Python.framework")
        let pyVerLib = ((fwDir as NSString).appendingPathComponent("Versions/3.12/lib"))
        env["DYLD_FRAMEWORK_PATH"] = fwDir
        env["DYLD_LIBRARY_PATH"] = pyVerLib
        // Prepend our site-packages + resources for package resolution
        let existingPyPath = env["PYTHONPATH"] ?? ""
        env["PYTHONPATH"] = [sitePackages, resources, (resources as NSString).appendingPathComponent("holographic-fs"), (resources as NSString).appendingPathComponent("services"), existingPyPath].joined(separator: ":")
        env["HOLO_ROOT"] = dataDir.path
        env["GRID_SIZE"] = env["GRID_SIZE"] ?? "64"
        env["HOLO_ALLOWED_ORIGINS"] = env["HOLO_ALLOWED_ORIGINS"] ?? "http://localhost:3000,http://localhost:5173,capacitor://localhost, null"
        // Prefer GPU path by preloading the module (optional)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        p.environment = env

        // Pipe to app log
        let outPipe = Pipe(); p.standardOutput = outPipe
        let errPipe = Pipe(); p.standardError = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { fh in
            if let s = String(data: fh.availableData, encoding: .utf8), !s.isEmpty { print("[uvicorn]", s, terminator: "") }
        }
        errPipe.fileHandleForReading.readabilityHandler = { fh in
            if let s = String(data: fh.availableData, encoding: .utf8), !s.isEmpty { fputs("[uvicorn-err] \(s)", stderr) }
        }

        try p.run()
        self.process = p
    }

    func stop() {
        guard let p = process else { return }
        if p.isRunning { p.terminate() }
        process = nil
    }
}
