import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'pages/workout_page.dart';

void main() {
  // Statusbar-Style setzen
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: Color(0xFF0A0A0A),
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );
  
  runApp(const CalimiroApp());
}

class CalimiroApp extends StatelessWidget {
  const CalimiroApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Calimiro SmartMirror',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF9CA3AF),
          surface: Color(0xFF1A1A1A),
          background: Color(0xFF0A0A0A),
        ),
        fontFamily: 'System',
      ),
      debugShowCheckedModeBanner: false,
      routes: {
        '/': (_) => const _Home(),
        '/workout': (_) => const WorkoutPage(),
      },
      initialRoute: '/',
    );
  }
}

class _Home extends StatefulWidget {
  const _Home();

  @override
  State<_Home> createState() => _HomeState();
}

class _HomeState extends State<_Home> with TickerProviderStateMixin {
  late AnimationController _fadeController;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    
    _fadeController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );
    
    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeOut,
    ));
    
    // Sanfte Animation starten
    Future.delayed(const Duration(milliseconds: 200), () {
      _fadeController.forward();
    });
  }

  @override
  void dispose() {
    _fadeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0A),
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        systemOverlayStyle: SystemUiOverlayStyle.light,
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF0F0F0F),
              Color(0xFF0A0A0A),
              Color(0xFF050505),
            ],
            stops: [0.0, 0.5, 1.0],
          ),
        ),
        child: SafeArea(
          child: FadeTransition(
            opacity: _fadeAnimation,
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 24.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 20),
                  
                  // Header Section
                  _buildHeader(),
                  
                  const SizedBox(height: 60),
                  
                  // Features List
                  _buildFeaturesList(),
                  
                  const SizedBox(height: 40),
                  
                  // Main CTA Button
                  _buildMainButton(context),
                  
                  const SizedBox(height: 32),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Minimal Logo
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFF1A1A1A).withOpacity(0.6),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: const Color(0xFF2A2A2A).withOpacity(0.8),
              width: 1,
            ),
          ),
          child: Icon(
            Icons.fitness_center_outlined,
            color: Colors.white.withOpacity(0.9),
            size: 24,
          ),
        ),
        
        const SizedBox(height: 32),
        
        // Title
        const Text(
          'Calimiro',
          style: TextStyle(
            color: Colors.white,
            fontSize: 36,
            fontWeight: FontWeight.w300,
            letterSpacing: -0.5,
          ),
        ),
        
        const SizedBox(height: 8),
        
        Text(
          'SmartMirror',
          style: TextStyle(
            color: Colors.white.withOpacity(0.6),
            fontSize: 18,
            fontWeight: FontWeight.w400,
            letterSpacing: 1.5,
          ),
        ),
        
        const SizedBox(height: 24),
        
        // Subtitle
        Text(
          'Intelligente Trainingsanalyse\ndurch KI-gestützte Bewegungserkennung',
          style: TextStyle(
            color: Colors.white.withOpacity(0.5),
            fontSize: 15,
            height: 1.6,
            fontWeight: FontWeight.w300,
          ),
        ),
      ],
    );
  }

  Widget _buildFeaturesList() {
    return Column(
      children: [
        _buildFeatureItem(
          title: 'MediaPipe Pose Detection',
          description: 'Hochpräzise Echtzeit-Körpererkennung für natürliche Bewegungsanalyse',
          delay: 200,
        ),
        
        const SizedBox(height: 24),
        
        _buildFeatureItem(
          title: 'Biomechanische Analyse',
          description: 'Intelligente Klassifikation von Push-ups, Pull-ups, Squats und Dips',
          delay: 400,
        ),
        
        const SizedBox(height: 24),
        
        _buildFeatureItem(
          title: 'Automatische Wiederholungszählung',
          description: 'Präzise Bewegungserkennung für akkurate Rep-Erfassung',
          delay: 600,
        ),
        
        const SizedBox(height: 24),
        
        _buildFeatureItem(
          title: 'Live Performance Feedback',
          description: 'Sofortige Rückmeldung zu Bewegungsqualität und Trainingsfortschritt',
          delay: 800,
        ),
      ],
    );
  }

  Widget _buildFeatureItem({
    required String title,
    required String description,
    required int delay,
  }) {
    return TweenAnimationBuilder<double>(
      duration: Duration(milliseconds: 800 + delay),
      tween: Tween(begin: 0.0, end: 1.0),
      curve: Curves.easeOutCubic,
      builder: (context, value, child) {
        return Transform.translate(
          offset: Offset(0, 20 * (1 - value)),
          child: Opacity(
            opacity: value,
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: const Color(0xFF1A1A1A).withOpacity(0.4),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: const Color(0xFF2A2A2A).withOpacity(0.6),
                  width: 1,
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                      height: 1.3,
                    ),
                  ),
                  
                  const SizedBox(height: 8),
                  
                  Text(
                    description,
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.6),
                      fontSize: 13,
                      fontWeight: FontWeight.w300,
                      height: 1.5,
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildMainButton(BuildContext context) {
    return TweenAnimationBuilder<double>(
      duration: const Duration(milliseconds: 1200),
      tween: Tween(begin: 0.0, end: 1.0),
      curve: Curves.easeOutCubic,
      builder: (context, value, child) {
        return Transform.translate(
          offset: Offset(0, 30 * (1 - value)),
          child: Opacity(
            opacity: value,
            child: Container(
              width: double.infinity,
              height: 56,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    const Color(0xFF2A2A2A).withOpacity(0.8),
                    const Color(0xFF1A1A1A).withOpacity(0.9),
                  ],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: const Color(0xFF3A3A3A).withOpacity(0.8),
                  width: 1,
                ),
              ),
              child: Material(
                color: Colors.transparent,
                child: InkWell(
                  borderRadius: BorderRadius.circular(16),
                  onTap: () => _navigateToWorkout(context),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 24),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.arrow_forward_rounded,
                          color: Colors.white.withOpacity(0.9),
                          size: 20,
                        ),
                        const SizedBox(width: 12),
                        const Text(
                          'Workout Session starten',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w400,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  void _navigateToWorkout(BuildContext context) {
    // Sanftes Haptic feedback
    HapticFeedback.lightImpact();
    
    // Navigation
    Navigator.of(context).pushNamed('/workout');
  }
}