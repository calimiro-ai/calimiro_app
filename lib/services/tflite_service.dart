// lib/services/tflite_service.dart
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:camera/camera.dart';

class TfliteService {
  Interpreter? _interpreter;
  bool _isProcessing = false;

  bool get isLoaded => _interpreter != null;
  bool get isProcessing => _isProcessing;

  Future<void> load() async {
    if (_interpreter != null) return;
    try {
      _interpreter = await Interpreter.fromAsset('assets/models/pose_classifier.tflite');
      print('Modell erfolgreich geladen');
    } catch (e) {
      print('Fehler beim Laden des Modells: $e');
      rethrow;
    }
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }

  // Korrigierte Hilfsfunktion: sichere Behandlung von leeren shapes
  dynamic _zerosForShape(List<int> shape, double value) {
    // Empty-Check (null-Check nicht nötig, da shape nie null ist)
    if (shape.isEmpty) return value;
    
    final len = shape.first;
    final rest = shape.sublist(1);
    return List.generate(len, (_) => _zerosForShape(rest, value));
  }

  // NEUE METHODE: Verwende echte Pose-Daten aus der ML Kit Pose Detection
  Future<Map<String, dynamic>?> runInferenceWithPoseData(List<List<double>> poseHistory) async {
    if (_interpreter == null) {
      throw StateError('Interpreter not loaded');
    }

    if (_isProcessing) {
      return null; // Skip wenn noch processing läuft
    }

    if (poseHistory.isEmpty) {
      throw ArgumentError('poseHistory darf nicht leer sein');
    }

    _isProcessing = true;

    try {
      // Debug: Input-Shape zur Laufzeit prüfen
      final inputTensor = _interpreter!.getInputTensor(0);
      print('Erwartete Input-Shape: ${inputTensor.shape}');
      
      // Erstelle [1, 30, 25] Format aus den echten Pose-Daten
      final input = _preparePoseDataForModel(poseHistory);
      print('Erstellte Pose-Daten-Shape: [${input.length}, ${input[0].length}, ${input[0][0].length}]');

      // Outputs vorbereiten
      final outputTensors = _interpreter!.getOutputTensors();
      final outputMap = <int, Object>{};
      
      for (int i = 0; i < outputTensors.length; i++) {
        final shape = outputTensors[i].shape;
        outputMap[i] = _zerosForShape(shape, 0.0);
      }

      print('Pose-Daten werden an Modell gesendet...');
      
      // Inferenz ausführen - verwende runForMultipleInputs für Multi-Output
      _interpreter!.runForMultipleInputs([input], outputMap);

      print('Pose-basierte Inferenz erfolgreich!');

      // Ergebnisse verarbeiten
      final output1 = outputMap[0] as List; // [1, 30, 1]
      final output2 = outputMap[1] as List; // [1, 5]

      return {
        'timestamp': DateTime.now().millisecondsSinceEpoch,
        'poseDataSource': 'ML Kit Pose Detection',
        'poseFrameCount': poseHistory.length,
        'poseData': output1[0], // 30x1 Pose-Daten
        'classificationScores': output2[0], // 5 Klassifikations-Scores
        'rawOutputs': outputMap,
      };

    } catch (e) {
      print('Fehler bei Pose-basierter Inferenz: $e');
      rethrow;
    } finally {
      _isProcessing = false;
    }
  }

  // Bereite echte Pose-Daten für das Modell vor
  List<List<List<double>>> _preparePoseDataForModel(List<List<double>> poseHistory) {
    // Das Modell erwartet [1, 30, 25]
    // poseHistory hat bereits die richtige Struktur: List<List<double>> mit je 25 Elementen
    
    final int requiredFrames = 30;
    final List<List<double>> preparedFrames = [];
    
    if (poseHistory.length >= requiredFrames) {
      // Verwende die letzten 30 Frames
      preparedFrames.addAll(poseHistory.sublist(poseHistory.length - requiredFrames));
    } else {
      // Padding mit Nullen für fehlende Frames (sollte normalerweise nicht passieren)
      final missingFrames = requiredFrames - poseHistory.length;
      
      // Erst die echten Daten
      preparedFrames.addAll(poseHistory);
      
      // Dann Padding mit Nullen
      for (int i = 0; i < missingFrames; i++) {
        preparedFrames.add(List.filled(25, 0.0));
      }
    }
    
    // Validierung: Sicherstellen, dass jeder Frame genau 25 Features hat
    for (int i = 0; i < preparedFrames.length; i++) {
      if (preparedFrames[i].length != 25) {
        print('Warnung: Frame $i hat ${preparedFrames[i].length} Features statt 25');
        // Auf 25 Features anpassen
        if (preparedFrames[i].length < 25) {
          preparedFrames[i].addAll(List.filled(25 - preparedFrames[i].length, 0.0));
        } else {
          preparedFrames[i] = preparedFrames[i].sublist(0, 25);
        }
      }
    }
    
    // Rückgabe im erwarteten Format: [1, 30, 25]
    return [preparedFrames];
  }

  // ALTE METHODE: Konvertiert CameraImage zu synthetischen Daten (Fallback)
  List<List<List<double>>> _preprocessCameraImage(CameraImage image) {
    // Das Modell erwartet [1, 30, 25] - das ist wahrscheinlich:
    // - Batch: 1
    // - Zeitschritte/Sequenz: 30
    // - Features per Zeitschritt: 25
    // KEIN 4D Bildformat!
    
    final width = image.width.toDouble();
    final height = image.height.toDouble();
    
    // Erstelle GENAU die erwartete Shape: [1, 30, 25]
    return [
      // Ein Sample im Batch: 30 Zeitschritte mit je 25 Features
      List.generate(30, (timeStep) {
        return List.generate(25, (feature) {
          // Einfache Feature-Extraktion basierend auf Bildgröße und Position
          final normalizedTime = timeStep / 29.0;
          final normalizedFeature = feature / 24.0;
          final imageFeature = (width + height) / 2000.0; // Normalisiert
          
          // Feature-Kombination: Position + Zeit + Bildinfo
          return (normalizedTime * 0.3 + 
                  normalizedFeature * 0.3 + 
                  imageFeature * 0.4).clamp(0.0, 1.0);
        });
      })
    ];
  }

  // ALTE METHODE: Direkte Kamera-Inferenz (jetzt hauptsächlich für Tests)
  Future<Map<String, dynamic>?> runInference(CameraImage image) async {
    if (_interpreter == null) {
      throw StateError('Interpreter not loaded');
    }

    if (_isProcessing) {
      return null; // Skip frame wenn noch processing läuft
    }

    _isProcessing = true;

    try {
      // Debug: Input-Shape zur Laufzeit prüfen
      final inputTensor = _interpreter!.getInputTensor(0);
      print('Erwartete Input-Shape: ${inputTensor.shape}');
      
      // Preprocessing des Kamerabilds für das erwartete Format
      final preprocessedData = _preprocessCameraImage(image);
      print('Erstellte Daten-Shape: [${preprocessedData.length}, ${preprocessedData[0].length}, ${preprocessedData[0][0].length}]');
      
      // Input direkt verwenden - GENAU die erwartete Shape [1, 30, 25]
      final input = preprocessedData; // Direkt verwenden, nicht extrahieren!

      // Outputs vorbereiten
      final outputTensors = _interpreter!.getOutputTensors();
      final outputMap = <int, Object>{};
      
      for (int i = 0; i < outputTensors.length; i++) {
        final shape = outputTensors[i].shape;
        outputMap[i] = _zerosForShape(shape, 0.0);
      }

      print('Input wird an Modell gesendet...');
      
      // Inferenz ausführen - verwende runForMultipleInputs für Multi-Output
      _interpreter!.runForMultipleInputs([input], outputMap);

      print('Inferenz erfolgreich!');

      // Ergebnisse verarbeiten
      final output1 = outputMap[0] as List; // [1, 30, 1]
      final output2 = outputMap[1] as List; // [1, 5]

      return {
        'timestamp': DateTime.now().millisecondsSinceEpoch,
        'imageSize': '${image.width}x${image.height}',
        'poseDataSource': 'Synthetic (Camera Image)',
        'poseData': output1[0], // 30x1 Pose-Daten (nicht 30x3!)
        'classificationScores': output2[0], // 5 Klassifikations-Scores
        'rawOutputs': outputMap,
      };

    } catch (e) {
      print('Fehler bei Inferenz: $e');
      rethrow;
    } finally {
      _isProcessing = false;
    }
  }

  // Einfacher „Smoke Test": run() mit Nullen -> gibt erstes Output-Sample zurück.
  Map<String, dynamic> runDry() {
    if (_interpreter == null) {
      throw StateError('Interpreter not loaded');
    }

    try {
      final inputTensors = _interpreter!.getInputTensors();
      final outputTensors = _interpreter!.getOutputTensors();
      
      print('=== TensorFlow Lite Model Dry Run ===');
      print('Input Tensors: ${inputTensors.length}');
      for (int i = 0; i < inputTensors.length; i++) {
        final tensor = inputTensors[i];
        print('  Input $i: ${tensor.shape} (${tensor.type})');
      }
      
      print('Output Tensors: ${outputTensors.length}');
      for (int i = 0; i < outputTensors.length; i++) {
        final tensor = outputTensors[i];
        print('  Output $i: ${tensor.shape} (${tensor.type})');
      }

      // Input mit Nullen befüllen - für alle Tensor-Konfigurationen
      final List<Object> inputs = [];
      for (final inputTensor in inputTensors) {
        inputs.add(_zerosForShape(inputTensor.shape, 0.0));
      }

      // Output-Map vorbereiten
      final outputMap = <int, Object>{};
      for (int i = 0; i < outputTensors.length; i++) {
        final shape = outputTensors[i].shape;
        outputMap[i] = _zerosForShape(shape, 0.0);
      }

      // Model ausführen
      if (inputTensors.length == 1 && outputTensors.length == 1) {
        // Single Input/Output
        final output = _zerosForShape(outputTensors.first.shape, 0.0);
        _interpreter!.run(inputs.first, output);
        outputMap[0] = output;
      } else if (inputTensors.length == 1 && outputTensors.length > 1) {
        // Single Input, Multiple Output
        _interpreter!.runForMultipleInputs([inputs.first], outputMap);
      } else if (inputTensors.length > 1) {
        // Multiple Inputs
        _interpreter!.runForMultipleInputs(inputs, outputMap);
      } else {
        throw StateError('Unsupported tensor configuration: ${inputs.length} inputs, ${outputTensors.length} outputs');
      }

      // Erste paar Werte aus jedem Output extrahieren für Preview
      final previewData = <String, dynamic>{};
      for (int i = 0; i < outputTensors.length; i++) {
        final output = outputMap[i];
        previewData['output_$i'] = _extractPreview(output, 5);
      }

      return {
        'inputShapes': inputTensors.map((t) => t.shape).toList(),
        'outputShapes': outputTensors.map((t) => t.shape).toList(),
        'outputCount': outputTensors.length,
        'preview': previewData,
        'status': 'success',
      };
      
    } catch (e) {
      print('Fehler in runDry: $e');
      print('Stack trace: ${StackTrace.current}');
      rethrow;
    }
  }

  // Hilfsmethode: Extrahiere Preview-Werte aus verschachtelten Listen
  dynamic _extractPreview(dynamic data, int maxItems) {
    if (data is List) {
      if (data.isEmpty) return [];
      if (data.first is List) {
        // Verschachtelte Liste - gehe tiefer
        return data.take(maxItems).map((item) => _extractPreview(item, maxItems)).toList();
      } else {
        // Einfache Liste - nehme erste paar Werte
        return data.take(maxItems).toList();
      }
    } else {
      return data;
    }
  }

  // Debug-Methode um Modell-Informationen zu erhalten
  Map<String, dynamic> getModelInfo() {
    if (_interpreter == null) {
      return {'error': 'Interpreter not loaded'};
    }

    try {
      final inputTensors = _interpreter!.getInputTensors();
      final outputTensors = _interpreter!.getOutputTensors();
      
      return {
        'inputTensorCount': inputTensors.length,
        'outputTensorCount': outputTensors.length,
        'inputShapes': inputTensors
            .asMap()
            .map((i, tensor) => MapEntry(i, tensor.shape)),
        'outputShapes': outputTensors
            .asMap()
            .map((i, tensor) => MapEntry(i, tensor.shape)),
        'inputTypes': inputTensors
            .asMap()
            .map((i, tensor) => MapEntry(i, tensor.type.toString())),
        'outputTypes': outputTensors
            .asMap()
            .map((i, tensor) => MapEntry(i, tensor.type.toString())),
      };
    } catch (e) {
      return {'error': 'Failed to get model info: $e'};
    }
  }
}