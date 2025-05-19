import 'package:flutter/material.dart';
import 'package:snipai/pages%20and%20logic/home.dart';
import 'package:snipai/pages%20and%20logic/results.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      initialRoute: '/',
      routes: {
        '/': (context) => const Home(),
        '/results': (context) {
          final args = ModalRoute.of(context)?.settings.arguments;
          if (args == null) {
            return const Home(); // Fallback to home if no arguments
          }
          return ResultsPage(
            videoData: args as Map<String, dynamic>,
          );
        },
      },
    );
  }
}