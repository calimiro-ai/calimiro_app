import 'package:tflite_flutter/tflite_flutter.dart';

class TfliteService {
  Interpreter? _interpreter;

  bool get isLoaded => _interpreter != null;

  Future<void> load() async {
    if (_interpreter != null) return;
    _interpreter = await Interpreter.fromAsset('assets/models/pose_classifier.tflite');
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }

  // Hilfsfunktion: erzeugt eine verschachtelte Liste im Tensor-Shape, gefüllt mit 0.0
  dynamic _zerosForShape(List<int> shape, double value) {
    if (shape.isEmpty) return value;
    final len = shape.first;
    final rest = shape.sublist(1);
    return List.generate(len, (_) => _zerosForShape(rest, value));
  }

  // Einfacher „Smoke Test“: run() mit Nullen -> gibt erstes Output-Sample zurück.
  Map<String, dynamic> runDry() {
    if (_interpreter == null) {
      throw StateError('Interpreter not loaded');
    }
    final iShape = _interpreter!.getInputTensor(0).shape;
    final oShape = _interpreter!.getOutputTensor(0).shape;

    final input = _zerosForShape(iShape, 0.0);
    final output = _zerosForShape(oShape, 0.0);

    _interpreter!.run(input, output);

    // flatten + ein paar Werte zeigen
    List<double> _flatten(dynamic x) {
      if (x is num) return [x.toDouble()];
      if (x is List) {
        return x.expand((e) => _flatten(e)).toList();
      }
      return [];
    }

    final flat = _flatten(output);
    final preview = flat.take(8).toList();

    return {
      'inputShape': iShape,
      'outputShape': oShape,
      'preview': preview, // erste 8 Werte
    };
  }
}