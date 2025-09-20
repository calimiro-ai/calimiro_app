// lib/pages/workout_page.dart
import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/tflite_service.dart';
import '../services/pose_mlkit_service.dart';
import '../services/mediapipe_exercise_analyzer.dart';

class WorkoutPage extends StatefulWidget {
  const WorkoutPage({super.key});

  @override
  State<WorkoutPage> createState() => _WorkoutPageState();
}

class _WorkoutPageState extends State<WorkoutPage> {
  CameraController? _controller;
  late final TfliteService _tflite;
  late final PoseMlkitService _poseService;
  late final MediaPipeExerciseAnalyzer _exerciseAnalyzer;
  String _status = 'Init...';
  String _lastOutput = '—';
  
  // Live-Stream-Variablen
  bool _isLiveAnalysisRunning = false;
  Timer? _analysisTimer;
  StreamSubscription<CameraImage>? _imageStreamSubscription;
  
  // Performance-Monitoring
  int _frameCount = 0;
  double _avgInferenceTime = 0;
  List<double> _inferenceTimes = [];

  // MediaPipe Analyse-Statistiken
  int _poseDetectionSuccessCount = 0;
  int _poseDetectionFailureCount = 0;
  int _mediaPipeAnalysisCount = 0;

  @override
  void initState() {
    super.initState();
    _tflite = TfliteService();
    _poseService = PoseMlkitService();
    _exerciseAnalyzer = MediaPipeExerciseAnalyzer();
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
        imageFormatGroup: ImageFormatGroup.nv21, // Bessere Kompatibilität mit ML Kit
      );
      
      await _controller!.initialize();
      
      if (!mounted) return; // Widget wurde disposed während Initialisierung
      
      print('Kamera erfolgreich initialisiert: ${_controller!.value.previewSize}');
      
    } catch (e) {
      print('Kamera-Initialisierungsfehler: $e');
      setState(() => _status = 'Kamera-Fehler: ${e.toString()}');
      return;
    }

    setState(() => _status = 'Lade Services...');
    try {
      // Pose Detection Service initialisieren
      await _poseService.init();
      
      // Optional: TensorFlow Lite Modell laden (für zusätzliche Analyse)
      try {
        await _tflite.load();
        print('TensorFlow Lite Modell optional geladen');
      } catch (e) {
        print('TensorFlow Lite nicht verfügbar (optional): $e');
      }
      
      if (!mounted) return; // Widget wurde disposed während Service-Laden
      
      setState(() => _status = 'Bereit');
      print('MediaPipe Setup erfolgreich abgeschlossen');
      
    } catch (e) {
      print('Service-Ladefehler: $e');
      setState(() => _status = 'Service-Ladefehler: ${e.toString()}');
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
    
    // Services schließen
    _tflite.close();
    await _poseService.close();
    
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
      _poseDetectionSuccessCount = 0;
      _poseDetectionFailureCount = 0;
      _mediaPipeAnalysisCount = 0;
      _inferenceTimes.clear();
      _exerciseAnalyzer.clearHistory(); // MediaPipe Analyzer zurücksetzen
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

  // Kamera-Rotation bestimmen
  InputImageRotation _getImageRotation(CameraDescription camera) {
    switch (camera.sensorOrientation) {
      case 90:
        return InputImageRotation.rotation90deg;
      case 180:
        return InputImageRotation.rotation180deg;
      case 270:
        return InputImageRotation.rotation270deg;
      default:
        return InputImageRotation.rotation0deg;
    }
  }

  // Echtzeit-Inferenz mit MediaPipe Pose-Analyse
  Future<void> _runLiveInference(CameraImage image) async {
    if (!mounted || !_isLiveAnalysisRunning) return;
    
    final stopwatch = Stopwatch()..start();
    
    try {
      // 1. Pose Detection: Extrahiere vollständige Pose über PoseMlkitService
      final rotation = _getImageRotation(_controller!.description);
      final pose = await _getPoseFromImage(image, rotation);
      
      if (pose == null) {
        _poseDetectionFailureCount++;
        
        if (mounted) {
          setState(() {
            _lastOutput = 'Keine Pose erkannt...\n\n'
                         'Pose Detection Status:\n'
                         '✓ Erfolg: $_poseDetectionSuccessCount\n'
                         '✗ Fehler: $_poseDetectionFailureCount\n\n'
                         'Tipps:\n'
                         '• Stellen Sie sicher, dass Sie vollständig im Bild sind\n'
                         '• Sorgen Sie für gute Beleuchtung\n'
                         '• Vermeiden Sie komplexe Hintergründe';
          });
        }
        return;
      }
      
      _poseDetectionSuccessCount++;
      
      // 2. MediaPipe Analyse der erkannten Pose
      final analysisResult = _exerciseAnalyzer.analyzePose(pose);
      
      _mediaPipeAnalysisCount++;
      stopwatch.stop();
      _inferenceTimes.add(stopwatch.elapsedMilliseconds.toDouble());
      
      // 3. UI aktualisieren
      if (mounted && _isLiveAnalysisRunning) {
        setState(() {
          _lastOutput = _formatMediaPipeResults(analysisResult, pose);
        });
      }
      
    } catch (e) {
      if (mounted) {
        setState(() {
          _lastOutput = 'MediaPipe Analyse Fehler: ${e.toString()}';
        });
      }
      print('MediaPipe Analyse Fehler: $e');
    }
  }

  // Formatiere MediaPipe Analyse-Ergebnisse für Anzeige
  String _formatMediaPipeResults(
    Map<String, dynamic> analysisResult,
    Pose pose,
  ) {
    final StringBuffer output = StringBuffer();
    
    // 1. Hauptergebnis: Erkannte Übung
    output.writeln('=== ÜBUNGS-ERKENNUNG ===');
    output.writeln('Erkannte Übung: ${analysisResult['exercise']}');
    output.writeln('Konfidenz: ${(analysisResult['confidence'] * 100).toStringAsFixed(1)}%');
    output.writeln('Wiederholungen: ${analysisResult['reps']}');
    output.writeln('Status: ${analysisResult['inMovement'] ? "🏃 In Bewegung" : "⏸ Ruhend"}');
    output.writeln('Grund: ${analysisResult['reason']}');
    
    // 2. Alle Übungsscores
    if (analysisResult['allScores'] is Map) {
      output.writeln('\n=== ANALYSE-SCORES ===');
      final scores = analysisResult['allScores'] as Map<String, double>;
      final sortedEntries = scores.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));
      
      for (final entry in sortedEntries) {
        final score = entry.value * 100;
        final marker = entry.key == analysisResult['exercise'] ? '🏆' : '  ';
        output.writeln('$marker ${entry.key}: ${score.toStringAsFixed(1)}%');
      }
    }
    
    // 3. Performance-Statistiken
    output.writeln('\n=== PERFORMANCE ===');
    output.writeln('Frames: $_frameCount | MediaPipe Analysen: $_mediaPipeAnalysisCount');
    output.writeln('∅ Analyse-Zeit: ${_avgInferenceTime.toStringAsFixed(1)}ms');
    
    // 4. Pose Detection Status
    final totalPoseAttempts = _poseDetectionSuccessCount + _poseDetectionFailureCount;
    final poseSuccessRate = totalPoseAttempts > 0 
        ? (_poseDetectionSuccessCount / totalPoseAttempts * 100) 
        : 0.0;
    
    output.writeln('\n=== POSE DETECTION ===');
    output.writeln('Erfolgsrate: ${poseSuccessRate.toStringAsFixed(1)}% (${_poseDetectionSuccessCount}/${totalPoseAttempts})');
    output.writeln('Erkannte Landmarks: ${pose.landmarks.length}');
    
    // 5. Bewegungsintensität
    if (analysisResult['movementIntensity'] != null) {
      final intensity = analysisResult['movementIntensity'] as double;
      output.writeln('\n=== BEWEGUNGSANALYSE ===');
      output.writeln('Bewegungsintensität: ${(intensity * 1000).toStringAsFixed(2)}');
      output.writeln('Bewegungsbalken: ${"█" * (intensity * 1000 * 20).clamp(0, 20).round()}');
    }
    
    // 6. Wichtige Landmarks (Debug)
    output.writeln('\n=== KÖRPER-LANDMARKS ===');
    final keyLandmarks = [
      'left_shoulder', 'right_shoulder',
      'left_elbow', 'right_elbow', 
      'left_wrist', 'right_wrist',
      'left_hip', 'right_hip',
      'left_knee', 'right_knee'
    ];
    
    final availableLandmarks = <String>[];
    for (final type in PoseLandmarkType.values) {
      if (pose.landmarks[type] != null) {
        final name = type.toString().split('.').last;
        if (keyLandmarks.any((key) => name.toLowerCase().contains(key.toLowerCase()))) {
          availableLandmarks.add(name);
        }
      }
    }
    
    output.writeln('Verfügbare Key-Points: ${availableLandmarks.length}/10');
    output.writeln('Details: ${availableLandmarks.take(5).join(", ")}${availableLandmarks.length > 5 ? "..." : ""}');
    
    // 7. Analysetipps
    output.writeln('\n=== TIPPS ===');
    if (analysisResult['confidence'] < 0.6) {
      output.writeln('• Führen Sie charakteristische Bewegungen für die Übung aus');
      output.writeln('• Stellen Sie sicher, dass Sie vollständig im Bild sind');
    }
    if (analysisResult['reps'] == 0 && _mediaPipeAnalysisCount > 30) {
      output.writeln('• Führen Sie vollständige Wiederholungen aus');
      output.writeln('• Bewegen Sie sich gleichmäßig und kontrolliert');
    }
    
    return output.toString();
  }

  // Hilfsmethode: Hole Pose über PoseMlkitService
  Future<Pose?> _getPoseFromImage(CameraImage image, InputImageRotation rotation) async {
    try {
      // Erstelle InputImage mit der privaten Methode des PoseService
      final Uint8List bytes = image.planes[0].bytes;
      
      final inputImageData = InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: InputImageFormat.nv21,
        bytesPerRow: image.planes[0].bytesPerRow,
      );

      final inputImage = InputImage.fromBytes(
        bytes: bytes,
        metadata: inputImageData,
      );
      
      // Verwende den PoseDetector direkt über eine öffentliche Methode
      return await _detectPoseFromInputImage(inputImage);
      
    } catch (e) {
      print('Fehler bei Pose Detection: $e');
      return null;
    }
  }

  // Hilfsmethode für Pose Detection
  Future<Pose?> _detectPoseFromInputImage(InputImage inputImage) async {
    try {
      // Erstelle temporären PoseDetector da wir keinen direkten Zugriff haben
      final detector = PoseDetector(
        options: PoseDetectorOptions(
          mode: PoseDetectionMode.stream,
          model: PoseDetectionModel.base,
        ),
      );
      
      final poses = await detector.processImage(inputImage);
      await detector.close();
      
      return poses.isNotEmpty ? poses.first : null;
    } catch (e) {
      print('Fehler bei PoseDetector: $e');
      return null;
    }
  }

  // Reset Wiederholungszähler
  void _resetReps() {
    _exerciseAnalyzer.resetReps();
    setState(() {
      _lastOutput = 'Wiederholungszähler zurückgesetzt!\n\n$_lastOutput';
    });
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

  // Debug-Test für TensorFlow (optional)
  Future<void> _runModelDry() async {
    try {
      setState(() => _status = 'TensorFlow Test...');
      
      if (!_tflite.isLoaded) {
        setState(() {
          _status = 'Bereit';
          _lastOutput = 'TensorFlow Lite Modell nicht geladen (optional für reine MediaPipe-Analyse)';
        });
        return;
      }
      
      final res = _tflite.runDry();
      setState(() {
        _status = 'Bereit';
        _lastOutput =
            'TensorFlow Test Ergebnisse:\n'
            'Input Shapes: ${res['inputShapes']}\n'
            'Output Shapes: ${res['outputShapes']}\n'
            'Output Count: ${res['outputCount']}\n'
            'Preview: ${res['preview']}\n\n'
            'Hinweis: Hauptklassifikation läuft über MediaPipe!';
      });
    } catch (e) {
      setState(() {
        _status = 'Bereit';
        _lastOutput = 'TensorFlow Test Fehler:\n${e.toString()}\n\n'
                     'Das ist OK - MediaPipe funktioniert unabhängig!';
      });
    }
  }

  // Debug-Informationen über MediaPipe und optional TensorFlow
  Future<void> _debugModelOutput() async {
    try {
      setState(() => _status = 'Analysiere System...');
      
      final StringBuffer debug = StringBuffer();
      
      debug.writeln('=== MEDIAPIPE SYSTEM STATUS ===');
      debug.writeln('Pose Detection Service: ✓ Aktiv');
      debug.writeln('Exercise Analyzer: ✓ Bereit');
      debug.writeln('Unterstützte Übungen: ${MediaPipeExerciseAnalyzer.exercises.join(", ")}');
      
      // TensorFlow optional
      debug.writeln('\n=== TENSORFLOW STATUS (OPTIONAL) ===');
      if (_tflite.isLoaded) {
        try {
          final modelInfo = _tflite.getModelInfo();
          debug.writeln('TensorFlow Status: ✓ Verfügbar');
          debug.writeln('Model Info: ${modelInfo.entries.map((e) => '${e.key}: ${e.value}').join('\n')}');
        } catch (e) {
          debug.writeln('TensorFlow Status: ⚠ Fehler beim Laden: $e');
        }
      } else {
        debug.writeln('TensorFlow Status: ❌ Nicht geladen (optional)');
      }
      
      debug.writeln('\n=== SYSTEM-EMPFEHLUNGEN ===');
      debug.writeln('• MediaPipe läuft vollständig unabhängig');
      debug.writeln('• Übungsklassifikation: Biomechanische Analyse');
      debug.writeln('• Wiederholungszählung: Bewegungsintensitäts-Tracking');
      debug.writeln('• Start mit "MediaPipe Start" für beste Ergebnisse');
      
      setState(() {
        _status = 'Bereit';
        _lastOutput = debug.toString();
      });
      
    } catch (e) {
      setState(() {
        _status = 'Bereit';
        _lastOutput = 'Debug-Analyse Fehler:\n${e.toString()}';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final ready = _controller?.value.isInitialized == true; // Keine TensorFlow Abhängigkeit mehr

    return Scaffold(
      appBar: AppBar(
        title: const Text('Workout Session'),
        backgroundColor: _isLiveAnalysisRunning ? Colors.green : null,
      ),
      body: Column(
        children: [
          // Kamera-Vorschau
          Expanded(
            flex: 3,
            child: Container(
              width: double.infinity,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(8),
              ),
              child: ready
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: CameraPreview(_controller!),
                    )
                  : Center(
                      child: Text(
                        _status,
                        style: Theme.of(context).textTheme.headlineSmall,
                        textAlign: TextAlign.center,
                      ),
                    ),
            ),
          ),
          
          const SizedBox(height: 16),
          
          // Steuerungsknöpfe
          Wrap(
            spacing: 8,
            runSpacing: 8,
            alignment: WrapAlignment.center,
            children: [
              ElevatedButton.icon(
                onPressed: ready && !_isLiveAnalysisRunning ? _startLiveAnalysis : null,
                icon: const Icon(Icons.play_arrow),
                label: const Text('MediaPipe Start'),
              ),
              ElevatedButton.icon(
                onPressed: _isLiveAnalysisRunning ? _stopLiveAnalysis : null,
                icon: const Icon(Icons.stop),
                label: const Text('Stoppen'),
                style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
              ),
              ElevatedButton.icon(
                onPressed: ready && _isLiveAnalysisRunning ? _resetReps : null,
                icon: const Icon(Icons.refresh),
                label: const Text('Reps Reset'),
              ),
              ElevatedButton.icon(
                onPressed: ready && !_isLiveAnalysisRunning ? _runModelDry : null,
                icon: const Icon(Icons.science),
                label: const Text('TF Test'),
              ),
              ElevatedButton.icon(
                onPressed: ready ? _debugModelOutput : null,
                icon: const Icon(Icons.analytics),
                label: const Text('Debug'),
              ),
            ],
          ),
          
          const SizedBox(height: 16),
          
          // Status und Ausgabe
          Expanded(
            flex: 2,
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey),
              ),
              child: SingleChildScrollView(
                child: Text(
                  _lastOutput,
                  style: const TextStyle(fontFamily: 'monospace'),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}