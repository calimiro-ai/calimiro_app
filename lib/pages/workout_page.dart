import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/tflite_service.dart';

class WorkoutPage extends StatefulWidget {
  const WorkoutPage({super.key});

  @override
  State<WorkoutPage> createState() => _WorkoutPageState();
}

class _WorkoutPageState extends State<WorkoutPage> {
  CameraController? _controller;
  late final TfliteService _tflite;
  String _status = 'Init...';
  String _lastOutput = '—';
  
  // Live-Stream-Variablen
  bool _isLiveAnalysisRunning = false;
  Timer? _analysisTimer;
  StreamSubscription<CameraImage>? _imageStreamSubscription;
  
  // Performance-Monitoring
  int _frameCount = 0;
  int _inferenceCount = 0;
  DateTime? _lastPerformanceReset;
  double _avgInferenceTime = 0;
  List<double> _inferenceTimes = [];

  @override
  void initState() {
    super.initState();
    _tflite = TfliteService();
    _lastPerformanceReset = DateTime.now();
    _setup();
  }

  Future<void> _setup() async {
    setState(() => _status = 'Prüfe Berechtigungen...');
    final ok = await _ensureCameraPermission();
    if (!ok) {
      setState(() => _status = 'Kamerazugriff verweigert.');
      return;
    }

    setState(() => _status = 'Initialisiere Kamera...');
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) {
        setState(() => _status = 'Keine Kamera gefunden.');
        return;
      }
      
      final back = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cams.first,
      );

      _controller = CameraController(
        back,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420, // Optimiert für Verarbeitung
      );
      
      await _controller!.initialize();
      
      if (!mounted) return; // Widget wurde disposed während Initialisierung
      
      print('Kamera erfolgreich initialisiert: ${_controller!.value.previewSize}');
      
    } catch (e) {
      print('Kamera-Initialisierungsfehler: $e');
      setState(() => _status = 'Kamera-Fehler: ${e.toString()}');
      return;
    }

    setState(() => _status = 'Lade Modell...');
    try {
      await _tflite.load();
      
      if (!mounted) return; // Widget wurde disposed während Modell-Laden
      
      setState(() => _status = 'Bereit');
      print('Setup erfolgreich abgeschlossen');
      
    } catch (e) {
      print('Modell-Ladefehler: $e');
      setState(() => _status = 'Modell-Ladefehler: ${e.toString()}');
    }
  }

  Future<bool> _ensureCameraPermission() async {
    final status = await Permission.camera.request();
    return status.isGranted;
  }

  @override
  void dispose() async {
    print('Disposing WorkoutPage...');
    
    // Live-Analyse stoppen falls aktiv
    if (_isLiveAnalysisRunning) {
      await _stopLiveAnalysis();
    }
    
    // Timer stoppen
    _analysisTimer?.cancel();
    _analysisTimer = null;
    
    // Image Stream Subscription aufräumen
    await _imageStreamSubscription?.cancel();
    _imageStreamSubscription = null;
    
    // Zusätzliche Wartezeit für sauberes Cleanup
    await Future.delayed(const Duration(milliseconds: 300));
    
    // Controller disposal
    try {
      await _controller?.dispose();
      _controller = null;
      print('CameraController disposed');
    } catch (e) {
      print('Fehler beim Disposing der Kamera: $e');
    }
    
    // TensorFlow Lite Service schließen
    _tflite.close();
    
    super.dispose();
    print('WorkoutPage disposal complete');
  }

  // Live-Analyse starten
  Future<void> _startLiveAnalysis() async {
    if (_isLiveAnalysisRunning || _controller == null || !_controller!.value.isInitialized) return;

    setState(() {
      _isLiveAnalysisRunning = true;
      _status = 'Live-Analyse läuft...';
      _frameCount = 0;
      _inferenceCount = 0;
      _lastPerformanceReset = DateTime.now();
      _inferenceTimes.clear();
    });

    try {
      // Warte kurz bevor Stream gestartet wird
      await Future.delayed(const Duration(milliseconds: 100));
      
      // Kamera-Stream starten
      await _controller!.startImageStream(_onCameraImage);
      
      // Performance-Timer (alle 2 Sekunden)
      _analysisTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
        _updatePerformanceStats();
      });

      print('Live-Analyse erfolgreich gestartet');

    } catch (e) {
      print('Fehler beim Starten der Live-Analyse: $e');
      setState(() {
        _status = 'Fehler beim Starten: ${e.toString()}';
        _isLiveAnalysisRunning = false;
      });
    }
  }

  // Live-Analyse stoppen
  Future<void> _stopLiveAnalysis() async {
    if (!_isLiveAnalysisRunning) return;

    print('Stoppe Live-Analyse...');

    setState(() {
      _isLiveAnalysisRunning = false;
      _status = 'Stoppe Analyse...';
    });

    try {
      // Timer zuerst stoppen
      _analysisTimer?.cancel();
      _analysisTimer = null;

      // Warte auf laufende Inferenzen
      int waitCount = 0;
      while (_tflite.isProcessing && waitCount < 10) {
        await Future.delayed(const Duration(milliseconds: 100));
        waitCount++;
      }

      // Kamera-Stream stoppen
      if (_controller?.value.isStreamingImages == true) {
        await _controller!.stopImageStream();
        print('Kamera-Stream gestoppt');
      }

      // Zusätzliche Wartezeit für sauberes Cleanup
      await Future.delayed(const Duration(milliseconds: 200));

      setState(() {
        _status = 'Bereit';
      });

      print('Live-Analyse erfolgreich gestoppt');

    } catch (e) {
      print('Fehler beim Stoppen: $e');
      setState(() {
        _status = 'Fehler beim Stoppen: ${e.toString()}';
      });
    }
  }

  // Callback für jedes Kamerabild
  void _onCameraImage(CameraImage image) {
    if (!_isLiveAnalysisRunning || !mounted) return;
    
    _frameCount++;
    
    // Inferenz nur alle paar Frames ausführen (Performance)
    if (_frameCount % 3 == 0 && !_tflite.isProcessing) {
      _runLiveInference(image);
    }
  }

  // Echtzeit-Inferenz auf Kamerabild
  Future<void> _runLiveInference(CameraImage image) async {
    if (!mounted || !_isLiveAnalysisRunning) return;
    
    final stopwatch = Stopwatch()..start();
    
    try {
      final result = await _tflite.runInference(image);
      
      if (result != null && mounted && _isLiveAnalysisRunning) {
        _inferenceCount++;
        stopwatch.stop();
        _inferenceTimes.add(stopwatch.elapsedMilliseconds.toDouble());
        
        // UI aktualisieren
        setState(() {
          _lastOutput = _formatLiveResults(result);
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _lastOutput = 'Live-Inferenz Fehler: ${e.toString()}';
        });
      }
      print('Live-Inferenz Fehler: $e');
    }
  }

  // Formatiere Live-Ergebnisse für Anzeige
  String _formatLiveResults(Map<String, dynamic> result) {
    final classificationScores = result['classificationScores'] as List;
    final poseData = result['poseData'] as List; // [30, 1] Format - verschachtelt!
    
    // Top-Klassifikation finden
    double maxScore = 0;
    int maxIndex = 0;
    for (int i = 0; i < classificationScores.length; i++) {
      if (classificationScores[i] > maxScore) {
        maxScore = classificationScores[i];
        maxIndex = i;
      }
    }

    final workoutClasses = ['Squats', 'Push-ups', 'Plank', 'Jumping Jacks', 'Rest'];
    final detectedClass = maxIndex < workoutClasses.length 
        ? workoutClasses[maxIndex] 
        : 'Unknown';

    // Pose-Daten richtig extrahieren - jedes Element ist eine Liste mit einem Wert
    final flatPoseData = poseData.map((item) {
      if (item is List && item.isNotEmpty) {
        return item[0] as double; // Extrahiere den Wert aus [wert]
      } else if (item is double) {
        return item; // Falls es bereits ein double ist
      } else {
        return 0.0; // Fallback
      }
    }).toList();

    return '''Live-Analyse:
Erkannte Übung: $detectedClass
Konfidenz: ${(maxScore * 100).toStringAsFixed(1)}%

Klassifikations-Scores:
${workoutClasses.asMap().entries.map((e) => 
  '${e.value}: ${(classificationScores[e.key] * 100).toStringAsFixed(1)}%'
).join('\n')}

Performance:
Frames: $_frameCount | Inferenzen: $_inferenceCount
Ø Inferenz-Zeit: ${_avgInferenceTime.toStringAsFixed(1)}ms

 Pose-Daten (erste 10):
${flatPoseData.take(10).map((value) => value.toStringAsFixed(3)).join(', ')}
''';
  }

  // Performance-Statistiken aktualisieren
  void _updatePerformanceStats() {
    if (_inferenceTimes.isNotEmpty) {
      _avgInferenceTime = _inferenceTimes.reduce((a, b) => a + b) / _inferenceTimes.length;
      
      // Nur die letzten 10 Zeiten behalten
      if (_inferenceTimes.length > 10) {
        _inferenceTimes = _inferenceTimes.sublist(_inferenceTimes.length - 10);
      }
    }
  }

  // Dry-Run für Test-Zwecke
  Future<void> _runModelDry() async {
    try {
      setState(() => _status = 'Inference (dry run)...');
      final res = _tflite.runDry();
      setState(() {
        _status = 'Bereit';
        _lastOutput =
            'OutputShapes: ${res['outputShapes']}\n'
            'OutputCount: ${res['outputCount']}\n'
            'Preview: ${res['preview']}';
      });
    } catch (e) {
      setState(() {
        _status = 'Fehler';
        _lastOutput = 'Fehler bei Model-Ausführung:\n${e.toString()}';
      });
    }
  }

  // Debug-Informationen anzeigen
  Future<void> _debugModelInfo() async {
    try {
      setState(() => _status = 'Lade Modell-Info...');
      final info = _tflite.getModelInfo();
      setState(() {
        _status = 'Bereit';
        _lastOutput = 'Modell-Info:\n${info.entries.map((e) => '${e.key}: ${e.value}').join('\n')}';
      });
    } catch (e) {
      setState(() {
        _status = 'Fehler';
        _lastOutput = 'Fehler beim Abrufen der Modell-Info:\n${e.toString()}';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final ready = _controller?.value.isInitialized == true && _tflite.isLoaded;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Workout Session'),
        backgroundColor: _isLiveAnalysisRunning ? Colors.green[100] : null,
      ),
      body: Column(
        children: [
          // Kamera-Preview
          AspectRatio(
            aspectRatio: ready ? _controller!.value.aspectRatio : 3 / 4,
            child: Stack(
              children: [
                ready
                    ? CameraPreview(_controller!)
                    : Container(
                        color: Colors.black12,
                        child: Center(child: Text(_status)),
                      ),
                // Live-Indikator
                if (_isLiveAnalysisRunning)
                  Positioned(
                    top: 10,
                    right: 10,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.red,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Container(
                            width: 8,
                            height: 8,
                            decoration: const BoxDecoration(
                              color: Colors.white,
                              shape: BoxShape.circle,
                            ),
                          ),
                          const SizedBox(width: 4),
                          const Text(
                            'LIVE',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          
          // Status und Steuerelemente
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Column(
              children: [
                Row(
                  children: [
                    Expanded(child: Text('Status: $_status')),
                  ],
                ),
                const SizedBox(height: 8),
                
                // Haupt-Steuerelemente
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: ready && !_isLiveAnalysisRunning 
                            ? _startLiveAnalysis 
                            : null,
                        icon: const Icon(Icons.play_arrow),
                        label: const Text('Live-Analyse starten'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green,
                          foregroundColor: Colors.white,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: _isLiveAnalysisRunning 
                            ? _stopLiveAnalysis 
                            : null,
                        icon: const Icon(Icons.stop),
                        label: const Text('Stoppen'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.red,
                          foregroundColor: Colors.white,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                
                // Test-Buttons
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: ready && !_isLiveAnalysisRunning 
                            ? _runModelDry 
                            : null,
                        icon: const Icon(Icons.science),
                        label: const Text('Test Run'),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: _tflite.isLoaded && !_isLiveAnalysisRunning 
                            ? _debugModelInfo 
                            : null,
                        icon: const Icon(Icons.info),
                        label: const Text('Debug'),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          
          // Ausgabe-Bereich
          Expanded(
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 12),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey[300]!),
              ),
              child: SingleChildScrollView(
                child: Text(
                  _lastOutput,
                  style: const TextStyle(
                    fontFamily: 'monospace', 
                    fontSize: 11,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}