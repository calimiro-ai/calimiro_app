// lib/services/pose_mlkit_service.dart
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

class PoseMlkitService {
  late final PoseDetector _detector;

  Future<void> init() async {
    _detector = PoseDetector(
      options: PoseDetectorOptions(
        mode: PoseDetectionMode.stream,
        model: PoseDetectionModel.base,
      ),
    );
  }

  Future<void> close() => _detector.close();

  // Reihenfolge der 24 Keypoints (entspricht eurem _relevantIds-Layout!)
  static const List<PoseLandmarkType> _order = [
    PoseLandmarkType.leftEye, PoseLandmarkType.rightEye,
    PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder,
    PoseLandmarkType.leftElbow, PoseLandmarkType.rightElbow,
    PoseLandmarkType.leftWrist, PoseLandmarkType.rightWrist,
    PoseLandmarkType.leftPinky, PoseLandmarkType.rightPinky,
    PoseLandmarkType.leftIndex, PoseLandmarkType.rightIndex,
    PoseLandmarkType.leftThumb, PoseLandmarkType.rightThumb,
    PoseLandmarkType.leftHip, PoseLandmarkType.rightHip,
    PoseLandmarkType.leftKnee, PoseLandmarkType.rightKnee,
    PoseLandmarkType.leftAnkle, PoseLandmarkType.rightAnkle,
    PoseLandmarkType.leftHeel, PoseLandmarkType.rightHeel,
    PoseLandmarkType.leftFootIndex, PoseLandmarkType.rightFootIndex,
  ];

  // Triplets in dieser Reihenfolge (identisch zu eurem Code)
  static const List<List<int>> triplets = [
    [2,0,3], [14,2,0], [15,3,1], [14,2,4], [15,3,5],
    [2,4,6], [3,5,7], [4,6,10], [5,7,11], [4,6,12], [5,7,13],
    [4,6,8], [5,7,9], [2,14,15], [3,15,14], [2,14,16], [3,15,17],
    [14,16,18], [15,17,19], [16,18,22], [17,19,23], [18,20,22],
    [19,21,23], [20,18,16], [21,19,17],
  ];

  // Haupt-API: liefert 25 Winkel in [0..1] (Winkel/π) oder null, wenn nichts erkannt
  Future<List<double>?> anglesFromCameraImage(
    CameraImage img, {
    required InputImageRotation rotation,
    bool mirrorX = false, // Frontkamera -> true
  }) async {
    final input = _toInputImage(img, rotation);
    final poses = await _detector.processImage(input);
    if (poses.isEmpty) return null;

    final pose = poses.first;
    final landmarks = pose.landmarks;

    // Normierte Offsets in unserer Reihenfolge
    final w = img.width.toDouble();
    final h = img.height.toDouble();
    final List<Offset> lm = [];
    for (final t in _order) {
      final l = landmarks[t];
      if (l == null) return null;
      double x = l.x / w;
      double y = l.y / h;
      if (mirrorX) x = 1.0 - x;
      lm.add(Offset(x, y));
    }

    double dist(Offset a, Offset b) {
      final dx = a.dx - b.dx, dy = a.dy - b.dy;
      return math.sqrt(dx*dx + dy*dy);
    }

    double angleAt(int a, int b, int c) {
      final A = lm[a], B = lm[b], C = lm[c];
      final ab = dist(A, B), bc = dist(B, C), ac = dist(A, C);
      if (ab == 0 || bc == 0) return 0.0;
      final cosv = ((ab*ab + bc*bc - ac*ac) / (2*ab*bc)).clamp(-1.0, 1.0);
      return math.acos(cosv) / math.pi; // [0..1]
    }

    final out = <double>[];
    for (final t in triplets) { out.add(angleAt(t[0], t[1], t[2])); }
    return out.length == 25 && !out.any((e) => e.isNaN) ? out : null;
  }

  // ---- CameraImage -> InputImage (ML Kit) ----
  InputImage _toInputImage(CameraImage image, InputImageRotation rotation) {
    // Verwende nur die Bytes vom ersten Plane für bessere Kompatibilität
    final Uint8List bytes = image.planes[0].bytes;
    
    final inputImageData = InputImageMetadata(
      size: Size(image.width.toDouble(), image.height.toDouble()),
      rotation: rotation,
      format: InputImageFormat.nv21, // Explizit NV21 setzen
      bytesPerRow: image.planes[0].bytesPerRow,
    );

    return InputImage.fromBytes(
      bytes: bytes,
      metadata: inputImageData,
    );
  }
}

// Helper: Rotation aus Grad
InputImageRotation rotationFromDegrees(int deg) {
  switch (deg % 360) {
    case 0: return InputImageRotation.rotation0deg;
    case 90: return InputImageRotation.rotation90deg;
    case 180: return InputImageRotation.rotation180deg;
    case 270: return InputImageRotation.rotation270deg;
    default: return InputImageRotation.rotation0deg;
  }
}