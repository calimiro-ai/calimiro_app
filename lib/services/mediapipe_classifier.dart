// lib/services/mediapipe_classifier.dart
import 'dart:math' as math;
import 'package:flutter/material.dart';

class MediaPipeClassifier {
  // Die 4 Übungen auf die das Modell trainiert ist
  static const List<String> workoutClasses = [
    'Push Ups',
    'Pull Ups', 
    'Squats',
    'Dips'
  ];

  // MediaPipe Pose Landmark-Indices (33 Punkte)
  static const Map<String, int> landmarks = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32,
  };

  // Klassifiziere Übung basierend auf Landmark-Sequenz
  static Map<String, dynamic> classifyExercise(List<List<List<double>>> landmarkSequence) {
    if (landmarkSequence.isEmpty) {
      return {
        'exercise': 'Unknown',
        'confidence': 0.0,
        'scores': List.filled(4, 0.0),
        'reason': 'No landmark data'
      };
    }

    // Nimm die letzte Sequenz für Analyse
    final currentSequence = landmarkSequence.last;
    
    if (currentSequence.length < 10) {
      return {
        'exercise': 'Unknown',
        'confidence': 0.0,
        'scores': List.filled(4, 0.0),
        'reason': 'Insufficient frames: ${currentSequence.length}'
      };
    }

    // Analysiere die letzten 10 Frames für Bewegungsmuster
    final analysisFrames = currentSequence.sublist(currentSequence.length - 10);
    
    final scores = [
      _analyzePushUps(analysisFrames),
      _analyzePullUps(analysisFrames),
      _analyzeSquats(analysisFrames),
      _analyzeDips(analysisFrames),
    ];

    // Finde beste Klassifikation
    double maxScore = 0.0;
    int maxIndex = 0;
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxIndex = i;
      }
    }

    return {
      'exercise': maxScore > 0.3 ? workoutClasses[maxIndex] : 'Unknown',
      'confidence': maxScore,
      'scores': scores,
      'reason': 'MediaPipe analysis'
    };
  }

  // Analysiere Push-Up Bewegungsmuster
  static double _analyzePushUps(List<List<double>> frames) {
    if (frames.length < 5) return 0.0;

    double score = 0.0;
    int validFrames = 0;

    for (final frame in frames) {
      if (frame.length < 33 * 3) continue; // Mindestens 33 3D-Punkte

      // Extrahiere relevante Punkte (x, y, z für jeden Landmark)
      final leftShoulder = _extractPoint(frame, landmarks['left_shoulder']!);
      final rightShoulder = _extractPoint(frame, landmarks['right_shoulder']!);
      final leftElbow = _extractPoint(frame, landmarks['left_elbow']!);
      final rightElbow = _extractPoint(frame, landmarks['right_elbow']!);
      final leftWrist = _extractPoint(frame, landmarks['left_wrist']!);
      final rightWrist = _extractPoint(frame, landmarks['right_wrist']!);
      final leftHip = _extractPoint(frame, landmarks['left_hip']!);
      final rightHip = _extractPoint(frame, landmarks['right_hip']!);

      if (_hasValidPoints([leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist, leftHip, rightHip])) {
        // Push-Up Kriterien:
        // 1. Arme sind aktiv (Ellbogen bewegen sich)
        // 2. Körper ist horizontal/geneigt
        // 3. Hände sind unter Schultern positioniert
        
        final armAngleLeft = _calculateAngle(leftShoulder, leftElbow, leftWrist);
        final armAngleRight = _calculateAngle(rightShoulder, rightElbow, rightWrist);
        final bodyAngle = _calculateBodyAngle(leftShoulder, rightShoulder, leftHip, rightHip);
        
        double frameScore = 0.0;
        
        // Arm-Winkel Bewertung (90° = optimal für Push-Up)
        if (armAngleLeft > 60 && armAngleLeft < 120) frameScore += 0.3;
        if (armAngleRight > 60 && armAngleRight < 120) frameScore += 0.3;
        
        // Körperhaltung Bewertung (horizontal bis leicht geneigt)
        if (bodyAngle > 160) frameScore += 0.4;
        
        score += frameScore;
        validFrames++;
      }
    }

    return validFrames > 0 ? (score / validFrames) : 0.0;
  }

  // Analysiere Pull-Up Bewegungsmuster
  static double _analyzePullUps(List<List<double>> frames) {
    if (frames.length < 5) return 0.0;

    double score = 0.0;
    int validFrames = 0;

    for (final frame in frames) {
      if (frame.length < 33 * 3) continue;

      final leftShoulder = _extractPoint(frame, landmarks['left_shoulder']!);
      final rightShoulder = _extractPoint(frame, landmarks['right_shoulder']!);
      final leftElbow = _extractPoint(frame, landmarks['left_elbow']!);
      final rightElbow = _extractPoint(frame, landmarks['right_elbow']!);
      final leftWrist = _extractPoint(frame, landmarks['left_wrist']!);
      final rightWrist = _extractPoint(frame, landmarks['right_wrist']!);

      if (_hasValidPoints([leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist])) {
        // Pull-Up Kriterien:
        // 1. Arme sind über dem Kopf
        // 2. Körper hängt vertikal
        // 3. Ellbogen bewegen sich nach unten/oben
        
        final armAngleLeft = _calculateAngle(leftShoulder, leftElbow, leftWrist);
        final armAngleRight = _calculateAngle(rightShoulder, rightElbow, rightWrist);
        
        double frameScore = 0.0;
        
        // Arme über Kopf Check (Y-Koordinate Handgelenk < Schulter)
        if (leftWrist.dy < leftShoulder.dy) frameScore += 0.25;
        if (rightWrist.dy < rightShoulder.dy) frameScore += 0.25;
        
        // Arm-Winkel für Pull-Up Bewegung
        if (armAngleLeft > 90 && armAngleLeft < 180) frameScore += 0.25;
        if (armAngleRight > 90 && armAngleRight < 180) frameScore += 0.25;
        
        score += frameScore;
        validFrames++;
      }
    }

    return validFrames > 0 ? (score / validFrames) : 0.0;
  }

  // Analysiere Squat Bewegungsmuster
  static double _analyzeSquats(List<List<double>> frames) {
    if (frames.length < 5) return 0.0;

    double score = 0.0;
    int validFrames = 0;

    for (final frame in frames) {
      if (frame.length < 33 * 3) continue;

      final leftHip = _extractPoint(frame, landmarks['left_hip']!);
      final rightHip = _extractPoint(frame, landmarks['right_hip']!);
      final leftKnee = _extractPoint(frame, landmarks['left_knee']!);
      final rightKnee = _extractPoint(frame, landmarks['right_knee']!);
      final leftAnkle = _extractPoint(frame, landmarks['left_ankle']!);
      final rightAnkle = _extractPoint(frame, landmarks['right_ankle']!);

      if (_hasValidPoints([leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle])) {
        // Squat Kriterien:
        // 1. Knie beugen sich
        // 2. Hüfte bewegt sich nach unten
        // 3. Aufrechte Körperhaltung
        
        final leftKneeAngle = _calculateAngle(leftHip, leftKnee, leftAnkle);
        final rightKneeAngle = _calculateAngle(rightHip, rightKnee, rightAnkle);
        
        double frameScore = 0.0;
        
        // Knie-Winkel Bewertung (90° = tiefer Squat)
        if (leftKneeAngle > 60 && leftKneeAngle < 140) frameScore += 0.4;
        if (rightKneeAngle > 60 && rightKneeAngle < 140) frameScore += 0.4;
        
        // Hüftposition prüfen (sollte zwischen Knien und Schultern sein)
        final avgHipY = (leftHip.dy + rightHip.dy) / 2;
        final avgKneeY = (leftKnee.dy + rightKnee.dy) / 2;
        if (avgHipY > avgKneeY) frameScore += 0.2; // Hüfte über Knien
        
        score += frameScore;
        validFrames++;
      }
    }

    return validFrames > 0 ? (score / validFrames) : 0.0;
  }

  // Analysiere Dip Bewegungsmuster
  static double _analyzeDips(List<List<double>> frames) {
    if (frames.length < 5) return 0.0;

    double score = 0.0;
    int validFrames = 0;

    for (final frame in frames) {
      if (frame.length < 33 * 3) continue;

      final leftShoulder = _extractPoint(frame, landmarks['left_shoulder']!);
      final rightShoulder = _extractPoint(frame, landmarks['right_shoulder']!);
      final leftElbow = _extractPoint(frame, landmarks['left_elbow']!);
      final rightElbow = _extractPoint(frame, landmarks['right_elbow']!);
      final leftWrist = _extractPoint(frame, landmarks['left_wrist']!);
      final rightWrist = _extractPoint(frame, landmarks['right_wrist']!);

      if (_hasValidPoints([leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist])) {
        // Dip Kriterien:
        // 1. Arme sind seitlich/hinter dem Körper
        // 2. Ellbogen bewegen sich nach außen
        // 3. Körper bewegt sich vertikal
        
        final armAngleLeft = _calculateAngle(leftShoulder, leftElbow, leftWrist);
        final armAngleRight = _calculateAngle(rightShoulder, rightElbow, rightWrist);
        
        double frameScore = 0.0;
        
        // Dip-spezifische Arm-Winkel
        if (armAngleLeft > 70 && armAngleLeft < 110) frameScore += 0.4;
        if (armAngleRight > 70 && armAngleRight < 110) frameScore += 0.4;
        
        // Handposition Check (sollten seitlich oder hinter den Schultern sein)
        if (leftWrist.dx < leftShoulder.dx) frameScore += 0.1;
        if (rightWrist.dx > rightShoulder.dx) frameScore += 0.1;
        
        score += frameScore;
        validFrames++;
      }
    }

    return validFrames > 0 ? (score / validFrames) : 0.0;
  }

  // Hilfsmethoden
  static Offset _extractPoint(List<double> frame, int landmarkIndex) {
    final baseIndex = landmarkIndex * 3; // x, y, z für jeden Punkt
    if (baseIndex + 1 >= frame.length) return const Offset(0, 0);
    return Offset(frame[baseIndex], frame[baseIndex + 1]);
  }

  static bool _hasValidPoints(List<Offset> points) {
    return points.every((p) => p.dx != 0.0 || p.dy != 0.0);
  }

  static double _calculateAngle(Offset a, Offset b, Offset c) {
    final ab = math.sqrt(math.pow(a.dx - b.dx, 2) + math.pow(a.dy - b.dy, 2));
    final bc = math.sqrt(math.pow(b.dx - c.dx, 2) + math.pow(b.dy - c.dy, 2));
    final ac = math.sqrt(math.pow(a.dx - c.dx, 2) + math.pow(a.dy - c.dy, 2));

    if (ab == 0 || bc == 0) return 0.0;

    final cosAngle = ((ab * ab + bc * bc - ac * ac) / (2 * ab * bc)).clamp(-1.0, 1.0);
    return math.acos(cosAngle) * 180 / math.pi; // Grad
  }

  static double _calculateBodyAngle(Offset leftShoulder, Offset rightShoulder, Offset leftHip, Offset rightHip) {
    final shoulderCenter = Offset((leftShoulder.dx + rightShoulder.dx) / 2, (leftShoulder.dy + rightShoulder.dy) / 2);
    final hipCenter = Offset((leftHip.dx + rightHip.dx) / 2, (leftHip.dy + rightHip.dy) / 2);
    
    final deltaY = shoulderCenter.dy - hipCenter.dy;
    final deltaX = shoulderCenter.dx - hipCenter.dx;
    
    if (deltaX == 0) return 90.0;
    
    final angle = math.atan(deltaY / deltaX) * 180 / math.pi;
    return (angle + 90).abs(); // Normalisiert auf 0-180°
  }

  // Debug-Methode: Analysiere rohe TensorFlow Lite Ausgaben
  static Map<String, dynamic> analyzeModelOutput(Map<String, dynamic> modelResult) {
    final output1 = modelResult['poseData'] as List?;
    final output2 = modelResult['classificationScores'] as List?;
    
    return {
      'output1_length': output1?.length ?? 0,
      'output1_sample': output1?.take(10).toList() ?? [],
      'output1_type': output1?.runtimeType.toString() ?? 'null',
      'output2_length': output2?.length ?? 0,
      'output2_sample': output2?.take(10).toList() ?? [],
      'output2_type': output2?.runtimeType.toString() ?? 'null',
      'rawOutputKeys': modelResult.keys.toList(),
    };
  }
}