// lib/services/mediapipe_exercise_analyzer.dart
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

class MediaPipeExerciseAnalyzer {
  // Die 4 Übungen die erkannt werden sollen
  static const List<String> exercises = [
    'Push Ups',
    'Pull Ups', 
    'Squats',
    'Dips'
  ];

  // Bewegungshistorie für zeitliche Analyse
  final List<Map<PoseLandmarkType, PoseLandmark>> _poseHistory = [];
  final List<double> _repSignal = [];
  int _detectedReps = 0;
  String _currentExercise = 'Unknown';
  double _currentConfidence = 0.0;
  
  // Bewegungszustand
  bool _inMovement = false;
  int _movementFrames = 0;
  
  static const int maxHistoryFrames = 15; // Kürzere Historie für bessere Reaktionszeit

  // Hauptmethode: Analysiere Pose und erkenne Übung
  Map<String, dynamic> analyzePose(Pose pose) {
    final landmarks = pose.landmarks;
    
    // Prüfe ob alle wichtigen Landmarks verfügbar sind
    if (!_hasRequiredLandmarks(landmarks)) {
      return {
        'exercise': 'Unknown',
        'confidence': 0.0,
        'reason': 'Insufficient landmarks',
        'reps': _detectedReps,
        'inMovement': false,
      };
    }

    // Füge zur Historie hinzu
    _poseHistory.add(Map.from(landmarks));
    if (_poseHistory.length > maxHistoryFrames) {
      _poseHistory.removeAt(0);
    }

    // Benötige mindestens 5 Frames für Analyse
    if (_poseHistory.length < 5) {
      return {
        'exercise': 'Collecting data...',
        'confidence': 0.0,
        'reason': 'Building history (${_poseHistory.length}/5)',
        'reps': _detectedReps,
        'inMovement': false,
      };
    }

    // Analysiere alle Übungen
    final exerciseScores = _analyzeAllExercises();
    
    // Finde beste Übung
    String bestExercise = 'Unknown';
    double bestScore = 0.0;
    
    for (final entry in exerciseScores.entries) {
      if (entry.value > bestScore) {
        bestScore = entry.value;
        bestExercise = entry.key;
      }
    }

    // Mindest-Konfidenz für Erkennung
    if (bestScore < 0.4) {
      bestExercise = 'Unknown';
      bestScore = 0.0;
    }

    // Glätte Ergebnisse (verhindere zu schnelle Wechsel)
    if (bestExercise != _currentExercise) {
      if (bestScore > _currentConfidence + 0.2) {
        _currentExercise = bestExercise;
        _currentConfidence = bestScore;
      }
    } else {
      // Gleiche Übung: Update Konfidenz langsam
      _currentConfidence = (_currentConfidence * 0.7) + (bestScore * 0.3);
    }

    // Bewegungserkennung für Reps
    final movementIntensity = _calculateMovementIntensity();
    _updateRepCount(movementIntensity);

    return {
      'exercise': _currentExercise,
      'confidence': _currentConfidence,
      'reason': 'MediaPipe analysis',
      'reps': _detectedReps,
      'inMovement': _inMovement,
      'movementIntensity': movementIntensity,
      'allScores': exerciseScores,
    };
  }

  // Analysiere alle Übungen und gib Scores zurück
  Map<String, double> _analyzeAllExercises() {
    return {
      'Push Ups': _analyzePushUps(),
      'Pull Ups': _analyzePullUps(),
      'Squats': _analyzeSquats(),
      'Dips': _analyzeDips(),
    };
  }

  // Push-Up Analyse basierend auf MediaPipe Landmarks
  double _analyzePushUps() {
    if (_poseHistory.length < 3) return 0.0;

    double totalScore = 0.0;
    int validFrames = 0;

    for (final landmarks in _poseHistory) {
      final leftShoulder = landmarks[PoseLandmarkType.leftShoulder];
      final rightShoulder = landmarks[PoseLandmarkType.rightShoulder];
      final leftElbow = landmarks[PoseLandmarkType.leftElbow];
      final rightElbow = landmarks[PoseLandmarkType.rightElbow];
      final leftWrist = landmarks[PoseLandmarkType.leftWrist];
      final rightWrist = landmarks[PoseLandmarkType.rightWrist];
      final leftHip = landmarks[PoseLandmarkType.leftHip];
      final rightHip = landmarks[PoseLandmarkType.rightHip];

      if (_allLandmarksValid([leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist, leftHip, rightHip])) {
        double frameScore = 0.0;

        // 1. Arm-Winkel (Push-Up Position)
        if (leftShoulder != null && leftElbow != null && leftWrist != null) {
          final leftArmAngle = _calculateAngle(
            Offset(leftShoulder.x, leftShoulder.y),
            Offset(leftElbow.x, leftElbow.y),
            Offset(leftWrist.x, leftWrist.y)
          );
          // Push-Up typische Arm-Winkel: 60-120 Grad
          if (leftArmAngle >= 60 && leftArmAngle <= 120) frameScore += 0.25;
        }
        
        if (rightShoulder != null && rightElbow != null && rightWrist != null) {
          final rightArmAngle = _calculateAngle(
            Offset(rightShoulder.x, rightShoulder.y),
            Offset(rightElbow.x, rightElbow.y),
            Offset(rightWrist.x, rightWrist.y)
          );
          if (rightArmAngle >= 60 && rightArmAngle <= 120) frameScore += 0.25;
        }

        // 2. Körperhaltung (gerade Linie Schulter-Hüfte)
        if (leftHip != null && rightHip != null && leftShoulder != null && rightShoulder != null) {
          final bodyAngle = _calculateBodyLineAngle(leftShoulder, rightShoulder, leftHip, rightHip);
          if (bodyAngle > 160) frameScore += 0.2; // Relativ gerade
        }

        // 3. Hände unter Schultern
        if (leftShoulder != null && rightShoulder != null && leftWrist != null && rightWrist != null) {
          final handsUnderShoulders = _areHandsUnderShoulders(leftShoulder, rightShoulder, leftWrist, rightWrist);
          if (handsUnderShoulders) frameScore += 0.2;
        }

        // 4. Horizontale Körperposition
        if (leftHip != null && rightHip != null && leftShoulder != null && rightShoulder != null) {
          final avgShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
          final avgHipY = (leftHip.y + rightHip.y) / 2;
          if ((avgShoulderY - avgHipY).abs() < 0.15) frameScore += 0.1; // Relativ horizontal
        }

        totalScore += frameScore;
        validFrames++;
      }
    }

    return validFrames > 0 ? totalScore / validFrames : 0.0;
  }

  // Squats Analyse
  double _analyzeSquats() {
    if (_poseHistory.length < 3) return 0.0;

    double totalScore = 0.0;
    int validFrames = 0;

    for (final landmarks in _poseHistory) {
      final leftHip = landmarks[PoseLandmarkType.leftHip];
      final rightHip = landmarks[PoseLandmarkType.rightHip];
      final leftKnee = landmarks[PoseLandmarkType.leftKnee];
      final rightKnee = landmarks[PoseLandmarkType.rightKnee];
      final leftAnkle = landmarks[PoseLandmarkType.leftAnkle];
      final rightAnkle = landmarks[PoseLandmarkType.rightAnkle];
      final leftShoulder = landmarks[PoseLandmarkType.leftShoulder];
      final rightShoulder = landmarks[PoseLandmarkType.rightShoulder];

      if (_allLandmarksValid([leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle, leftShoulder, rightShoulder])) {
        double frameScore = 0.0;

        // 1. Knie-Winkel (Squat-typisch: 70-140 Grad)
        if (leftHip != null && leftKnee != null && leftAnkle != null) {
          final leftKneeAngle = _calculateAngle(
            Offset(leftHip.x, leftHip.y),
            Offset(leftKnee.x, leftKnee.y),
            Offset(leftAnkle.x, leftAnkle.y)
          );
          if (leftKneeAngle >= 70 && leftKneeAngle <= 140) frameScore += 0.3;
        }
        
        if (rightHip != null && rightKnee != null && rightAnkle != null) {
          final rightKneeAngle = _calculateAngle(
            Offset(rightHip.x, rightHip.y),
            Offset(rightKnee.x, rightKnee.y),
            Offset(rightAnkle.x, rightAnkle.y)
          );
          if (rightKneeAngle >= 70 && rightKneeAngle <= 140) frameScore += 0.3;
        }

        // 2. Aufrechte Körperhaltung
        if (leftShoulder != null && rightShoulder != null && leftHip != null && rightHip != null) {
          final avgShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
          final avgHipY = (leftHip.y + rightHip.y) / 2;
          if (avgShoulderY < avgHipY) frameScore += 0.2; // Schultern über Hüfte
        }

        // 3. Füße am Boden (Knöchel sollten relativ stabil sein)
        if (leftAnkle != null && rightAnkle != null && leftHip != null && rightHip != null) {
          final avgAnkleY = (leftAnkle.y + rightAnkle.y) / 2;
          final avgHipY = (leftHip.y + rightHip.y) / 2;
          if (avgAnkleY > avgHipY) frameScore += 0.2; // Füße unter Hüfte
        }

        totalScore += frameScore;
        validFrames++;
      }
    }

    return validFrames > 0 ? totalScore / validFrames : 0.0;
  }

  // Pull-Ups Analyse
  double _analyzePullUps() {
    if (_poseHistory.length < 3) return 0.0;

    double totalScore = 0.0;
    int validFrames = 0;

    for (final landmarks in _poseHistory) {
      final leftShoulder = landmarks[PoseLandmarkType.leftShoulder];
      final rightShoulder = landmarks[PoseLandmarkType.rightShoulder];
      final leftElbow = landmarks[PoseLandmarkType.leftElbow];
      final rightElbow = landmarks[PoseLandmarkType.rightElbow];
      final leftWrist = landmarks[PoseLandmarkType.leftWrist];
      final rightWrist = landmarks[PoseLandmarkType.rightWrist];

      if (_allLandmarksValid([leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist])) {
        double frameScore = 0.0;

        // 1. Arme über Kopf
        if (leftWrist != null && leftShoulder != null) {
          if (leftWrist.y < leftShoulder.y) frameScore += 0.25;
        }
        if (rightWrist != null && rightShoulder != null) {
          if (rightWrist.y < rightShoulder.y) frameScore += 0.25;
        }

        // 2. Arm-Winkel für Pull-Up (typisch: 90-180 Grad)
        if (leftShoulder != null && leftElbow != null && leftWrist != null) {
          final leftArmAngle = _calculateAngle(
            Offset(leftShoulder.x, leftShoulder.y),
            Offset(leftElbow.x, leftElbow.y),
            Offset(leftWrist.x, leftWrist.y)
          );
          if (leftArmAngle >= 90 && leftArmAngle <= 180) frameScore += 0.25;
        }
        
        if (rightShoulder != null && rightElbow != null && rightWrist != null) {
          final rightArmAngle = _calculateAngle(
            Offset(rightShoulder.x, rightShoulder.y),
            Offset(rightElbow.x, rightElbow.y),
            Offset(rightWrist.x, rightWrist.y)
          );
          if (rightArmAngle >= 90 && rightArmAngle <= 180) frameScore += 0.25;
        }

        totalScore += frameScore;
        validFrames++;
      }
    }

    return validFrames > 0 ? totalScore / validFrames : 0.0;
  }

  // Dips Analyse
  double _analyzeDips() {
    if (_poseHistory.length < 3) return 0.0;

    double totalScore = 0.0;
    int validFrames = 0;

    for (final landmarks in _poseHistory) {
      final leftShoulder = landmarks[PoseLandmarkType.leftShoulder];
      final rightShoulder = landmarks[PoseLandmarkType.rightShoulder];
      final leftElbow = landmarks[PoseLandmarkType.leftElbow];
      final rightElbow = landmarks[PoseLandmarkType.rightElbow];
      final leftWrist = landmarks[PoseLandmarkType.leftWrist];
      final rightWrist = landmarks[PoseLandmarkType.rightWrist];

      if (_allLandmarksValid([leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist])) {
        double frameScore = 0.0;

        // 1. Dip-typische Arm-Winkel (70-110 Grad)
        if (leftShoulder != null && leftElbow != null && leftWrist != null) {
          final leftArmAngle = _calculateAngle(
            Offset(leftShoulder.x, leftShoulder.y),
            Offset(leftElbow.x, leftElbow.y),
            Offset(leftWrist.x, leftWrist.y)
          );
          if (leftArmAngle >= 70 && leftArmAngle <= 110) frameScore += 0.4;
        }
        
        if (rightShoulder != null && rightElbow != null && rightWrist != null) {
          final rightArmAngle = _calculateAngle(
            Offset(rightShoulder.x, rightShoulder.y),
            Offset(rightElbow.x, rightElbow.y),
            Offset(rightWrist.x, rightWrist.y)
          );
          if (rightArmAngle >= 70 && rightArmAngle <= 110) frameScore += 0.4;
        }

        // 2. Hände seitlich/hinter Schultern
        if (leftWrist != null && leftShoulder != null && rightWrist != null && rightShoulder != null) {
          if (leftWrist.x < leftShoulder.x - 0.05) frameScore += 0.1;
          if (rightWrist.x > rightShoulder.x + 0.05) frameScore += 0.1;
        }

        totalScore += frameScore;
        validFrames++;
      }
    }

    return validFrames > 0 ? totalScore / validFrames : 0.0;
  }

  // Hilfsmethoden
  bool _hasRequiredLandmarks(Map<PoseLandmarkType, PoseLandmark> landmarks) {
    final required = [
      PoseLandmarkType.leftShoulder,
      PoseLandmarkType.rightShoulder,
      PoseLandmarkType.leftElbow,
      PoseLandmarkType.rightElbow,
      PoseLandmarkType.leftWrist,
      PoseLandmarkType.rightWrist,
      PoseLandmarkType.leftHip,
      PoseLandmarkType.rightHip,
    ];

    return required.every((type) => landmarks[type] != null);
  }

  bool _allLandmarksValid(List<PoseLandmark?> landmarks) {
    return landmarks.every((landmark) => landmark != null);
  }

  double _calculateAngle(Offset a, Offset b, Offset c) {
    final ab = math.sqrt(math.pow(a.dx - b.dx, 2) + math.pow(a.dy - b.dy, 2));
    final bc = math.sqrt(math.pow(b.dx - c.dx, 2) + math.pow(b.dy - c.dy, 2));
    final ac = math.sqrt(math.pow(a.dx - c.dx, 2) + math.pow(a.dy - c.dy, 2));

    if (ab == 0 || bc == 0) return 0.0;

    final cosAngle = ((ab * ab + bc * bc - ac * ac) / (2 * ab * bc)).clamp(-1.0, 1.0);
    return math.acos(cosAngle) * 180 / math.pi; // Grad
  }

  double _calculateBodyLineAngle(PoseLandmark? leftShoulder, PoseLandmark? rightShoulder, PoseLandmark? leftHip, PoseLandmark? rightHip) {
    if (leftShoulder == null || rightShoulder == null || leftHip == null || rightHip == null) {
      return 0.0;
    }
    
    final shoulderCenter = Offset((leftShoulder.x + rightShoulder.x) / 2, (leftShoulder.y + rightShoulder.y) / 2);
    final hipCenter = Offset((leftHip.x + rightHip.x) / 2, (leftHip.y + rightHip.y) / 2);
    
    final deltaY = shoulderCenter.dy - hipCenter.dy;
    final deltaX = shoulderCenter.dx - hipCenter.dx;
    
    if (deltaX == 0) return 90.0;
    
    final angle = math.atan(deltaY / deltaX) * 180 / math.pi;
    return (angle + 90).abs();
  }

  bool _areHandsUnderShoulders(PoseLandmark? leftShoulder, PoseLandmark? rightShoulder, PoseLandmark? leftWrist, PoseLandmark? rightWrist) {
    if (leftShoulder == null || rightShoulder == null || leftWrist == null || rightWrist == null) {
      return false;
    }
    
    final shoulderWidth = (rightShoulder.x - leftShoulder.x).abs();
    final wristWidth = (rightWrist.x - leftWrist.x).abs();
    final wristCenterX = (leftWrist.x + rightWrist.x) / 2;
    final shoulderCenterX = (leftShoulder.x + rightShoulder.x) / 2;
    
    return (wristCenterX - shoulderCenterX).abs() < 0.1 && wristWidth <= shoulderWidth * 1.2;
  }

  double _calculateMovementIntensity() {
    if (_poseHistory.length < 3) return 0.0;

    double totalMovement = 0.0;
    final keyPoints = [
      PoseLandmarkType.leftShoulder,
      PoseLandmarkType.rightShoulder,
      PoseLandmarkType.leftElbow,
      PoseLandmarkType.rightElbow,
      PoseLandmarkType.leftWrist,
      PoseLandmarkType.rightWrist,
    ];

    for (int i = 1; i < _poseHistory.length; i++) {
      for (final pointType in keyPoints) {
        final prev = _poseHistory[i-1][pointType];
        final curr = _poseHistory[i][pointType];
        
        if (prev != null && curr != null) {
          final distance = math.sqrt(
            math.pow(curr.x - prev.x, 2) + math.pow(curr.y - prev.y, 2)
          );
          totalMovement += distance;
        }
      }
    }

    return totalMovement / (_poseHistory.length - 1);
  }

  void _updateRepCount(double movementIntensity) {
    _repSignal.add(movementIntensity);
    
    // Behalte nur letzte 30 Werte
    if (_repSignal.length > 30) {
      _repSignal.removeAt(0);
    }

    // Bewegungserkennung
    final isMoving = movementIntensity > 0.02;
    
    if (isMoving && !_inMovement) {
      _inMovement = true;
      _movementFrames = 0;
    } else if (!isMoving && _inMovement) {
      if (_movementFrames > 5) { // Mindestdauer für eine Rep
        _detectedReps++;
      }
      _inMovement = false;
      _movementFrames = 0;
    }

    if (_inMovement) {
      _movementFrames++;
    }
  }

  void resetReps() {
    _detectedReps = 0;
    _repSignal.clear();
    _inMovement = false;
    _movementFrames = 0;
  }

  void clearHistory() {
    _poseHistory.clear();
    _repSignal.clear();
    _detectedReps = 0;
    _inMovement = false;
    _movementFrames = 0;
    _currentExercise = 'Unknown';
    _currentConfidence = 0.0;
  }
}