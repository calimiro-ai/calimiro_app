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

  @override
  void initState() {
    super.initState();
    _tflite = TfliteService();
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
      );
      await _controller!.initialize();
    } catch (e) {
      setState(() => _status = 'Kamera-Fehler: ${e.toString()}');
      return;
    }

    setState(() => _status = 'Lade Modell...');
    try {
      await _tflite.load();
      setState(() => _status = 'Bereit');
    } catch (e) {
      setState(() => _status = 'Modell-Ladefehler: ${e.toString()}');
    }
  }

  Future<bool> _ensureCameraPermission() async {
    final status = await Permission.camera.request();
    return status.isGranted;
  }

  @override
  void dispose() {
    _controller?.dispose();
    _tflite.close();
    super.dispose();
  }

  Future<void> _runModelDry() async {
    try {
      setState(() => _status = 'Inference (dry run)...');
      final res = _tflite.runDry();
      setState(() {
        _status = 'Bereit';
        _lastOutput =
            'OutputShape: ${res['outputShape']}\nPreview: ${res['preview']}';
      });
    } catch (e) {
      setState(() {
        _status = 'Fehler';
        _lastOutput = 'Fehler bei Model-Ausführung:\n${e.toString()}';
      });
    }
  }

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
      appBar: AppBar(title: const Text('Workout Session')),
      body: Column(
        children: [
          AspectRatio(
            aspectRatio: ready ? _controller!.value.aspectRatio : 3 / 4,
            child: ready
                ? CameraPreview(_controller!)
                : Container(
                    color: Colors.black12,
                    child: Center(child: Text(_status)),
                  ),
          ),
          const SizedBox(height: 12),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Column(
              children: [
                Row(
                  children: [
                    Expanded(child: Text('Status: $_status')),
                    FilledButton.icon(
                      onPressed: ready ? _runModelDry : null,
                      icon: const Icon(Icons.play_arrow),
                      label: const Text('Run Model'),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: _tflite.isLoaded ? _debugModelInfo : null,
                        icon: const Icon(Icons.info),
                        label: const Text('Debug Model Info'),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(12),
              child: Text(
                _lastOutput,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
              ),
            ),
          ),
        ],
      ),
    );
  }
}