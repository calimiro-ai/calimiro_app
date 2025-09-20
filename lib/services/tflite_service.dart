import 'package:tflite_flutter/tflite_flutter.dart';

class TfliteService {
  Interpreter? _interpreter;

  bool get isLoaded => _interpreter != null;

  Future<void> load() async {
    if (_interpreter != null) return;
    try {
      _interpreter = await Interpreter.fromAsset('assets/models/pose_classifier.tflite');
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

  // Einfacher „Smoke Test": run() mit Nullen -> gibt erstes Output-Sample zurück.
  Map<String, dynamic> runDry() {
    if (_interpreter == null) {
      throw StateError('Interpreter not loaded');
    }

    try {
      // Alle Input- und Output-Tensoren abrufen
      final inputTensors = _interpreter!.getInputTensors();
      final outputTensors = _interpreter!.getOutputTensors();
      
      print('Input tensors count: ${inputTensors.length}');
      print('Output tensors count: ${outputTensors.length}');

      // Input-Daten erstellen (für alle Input-Tensoren)
      final inputs = <Object>[];
      for (int i = 0; i < inputTensors.length; i++) {
        final shape = inputTensors[i].shape;
        print('Input $i shape: $shape');
        
        if (shape.isEmpty) {
          throw StateError('Input tensor $i shape is empty: $shape');
        }
        
        inputs.add(_zerosForShape(shape, 0.0));
      }

      // Output-Daten erstellen (für alle Output-Tensoren)
      final outputs = <dynamic>[];
      for (int i = 0; i < outputTensors.length; i++) {
        final shape = outputTensors[i].shape;
        print('Output $i shape: $shape');
        
        if (shape.isEmpty) {
          throw StateError('Output tensor $i shape is empty: $shape');
        }
        
        outputs.add(_zerosForShape(shape, 0.0));
      }

      print('Input data created for ${inputs.length} tensors');
      print('Output data created for ${outputs.length} tensors');

      // Für Single-Input-Modell: direkt den ersten Input verwenden
      // Für Multi-Output-Modell: Map mit Output-Indices verwenden
      if (inputs.length == 1 && outputs.length > 1) {
        // Single Input, Multiple Outputs - korrigierter Typ
        final outputMap = <int, Object>{};
        for (int i = 0; i < outputs.length; i++) {
          outputMap[i] = outputs[i];
        }
        _interpreter!.runForMultipleInputs([inputs[0]], outputMap);
        
        // Preview vom ersten Output
        List<double> _flatten(dynamic x) {
          if (x is num) return [x.toDouble()];
          if (x is List) {
            return x.expand((e) => _flatten(e)).toList();
          }
          return [];
        }

        final flat = _flatten(outputs[0]);
        final preview = flat.take(8).toList();

        return {
          'inputShapes': inputTensors.map((t) => t.shape).toList(),
          'outputShapes': outputTensors.map((t) => t.shape).toList(),
          'preview': preview,
          'outputCount': outputs.length,
        };
      } else {
        // Fallback für andere Konfigurationen
        throw StateError('Unsupported tensor configuration: ${inputs.length} inputs, ${outputs.length} outputs');
      }
      
    } catch (e) {
      print('Fehler in runDry: $e');
      print('Stack trace: ${StackTrace.current}');
      rethrow;
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