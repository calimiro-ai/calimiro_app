import 'package:flutter/material.dart';
import 'pages/workout_page.dart';

void main() {
  runApp(const CalimiroApp());
}

class CalimiroApp extends StatelessWidget {
  const CalimiroApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Calimiro SmartMirror',
      theme: ThemeData(useMaterial3: true),
      debugShowCheckedModeBanner: false,
      routes: {
        '/': (_) => const _Home(),
        '/workout': (_) => const WorkoutPage(),
      },
      initialRoute: '/',
    );
  }
}

class _Home extends StatelessWidget {
  const _Home(); 

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Calimiro Home')),
      body: Center(
        child: FilledButton(
          onPressed: () => Navigator.pushNamed(context, '/workout'),
          child: const Text('Start Workout Session'),
        ),
      ),
    );
  }
}