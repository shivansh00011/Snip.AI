import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  final List<Map<String, dynamic>> features = [
    {
      'icon': Icons.search,
      'title': 'Transcribe',
      'description': 'Extract spoken words from any YouTube video.',
    },
    {
      'icon': Icons.content_cut,
      'title': 'Summarize',
      'description': 'Get concise breakdowns of long content.',
    },
    {
      'icon': Icons.smart_toy,
      'title': 'Ask Anything',
      'description': "Chat with your video like it's GPT-4 enabled.",
    },
  ];

  final TextEditingController _linkController = TextEditingController();
  bool _isLoading = false;

  Future<void> _handleGenerate() async {
    if (_linkController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a YouTube link')),
      );
      return;
    }

    // Navigate to results page immediately with loading state
    if (mounted) {
      Navigator.pushNamed(
        context,
        '/results',
        arguments: {
          'isLoading': true,
          'youtube_url': _linkController.text,
          'title': 'Processing...',
          'duration': 0,
        },
      );
    }

    try {
      final response = await http.post(
        Uri.parse('http://127.0.0.1:8000/transcribe'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'youtube_url': _linkController.text}),
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = json.decode(response.body) as Map<String, dynamic>;
        print('Backend response data: $responseData');
        
        // Extract session ID - try different possible field names
        String? sessionId;
        if (responseData.containsKey('transcript_id')) {
          sessionId = responseData['transcript_id']?.toString();
          print('Found session ID in transcript_id: $sessionId');
        } else if (responseData.containsKey('session_id')) {
          sessionId = responseData['session_id']?.toString();
          print('Found session ID in session_id: $sessionId');
        } else {
          // Try to find any field that looks like a UUID
          for (var key in responseData.keys) {
            final value = responseData[key]?.toString();
            if (value != null && value.contains('-')) {
              sessionId = value;
              print('Found potential session ID in $key: $sessionId');
              break;
            }
          }
        }
        
        if (sessionId == null) {
          print('ERROR: No session ID found in backend response');
          print('Available keys in response: ${responseData.keys.toList()}');
          print('Full response data: $responseData');
          if (mounted) {
            Navigator.pop(context); // Go back to home page
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Error: No session ID received from server')),
            );
          }
          return;
        }
        
        // Update the results page with the response data
        if (mounted) {
          final args = {
            'session_id': sessionId,
            'transcript_id': sessionId,
            'title': responseData['title']?.toString() ?? 'Video Title',
            'duration': (responseData['duration'] as num?)?.toInt() ?? 0,
            'thumbnail_url': responseData['thumbnail_url']?.toString(),
            'transcript': responseData['transcript']?.toString(),
            'summary': responseData['summary']?.toString(),
            'isLoading': false,
            'youtube_url': _linkController.text,
          };
          
          print('Navigating to results with args: $args');
          Navigator.pushReplacementNamed(
            context,
            '/results',
            arguments: args,
          );
        }
      } else {
        if (mounted) {
          Navigator.pop(context); // Go back to home page
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Error processing the video')),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        Navigator.pop(context); // Go back to home page
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Error connecting to server')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        height: double.infinity,
        width: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(colors: [
            const Color.fromARGB(255, 6, 48, 81),
            Colors.black
          ],
          begin: Alignment.centerLeft,
          end: Alignment.centerRight
          )
        ),
        child: Padding(
          padding: const EdgeInsets.only(left: 35, right: 35),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 40,),
              Center(child: Text("Snip.AI",style: GoogleFonts.montserrat(fontSize: 65, fontWeight: FontWeight.bold, color: Colors.white),)),
              const SizedBox(height: 12,),
              Center(child: Text("Transcribe, summarize, and ask — all from one link.", style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.w400, color: const Color.fromARGB(255, 213, 212, 212)),),),
              const SizedBox(height: 40,),
               Center(
                 child: Padding(
                   padding: const EdgeInsets.only(left: 18),
                   child: ClipRRect(
                             borderRadius: BorderRadius.circular(20),
                             child: BackdropFilter(
                               filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                               child: Container(
                                 height: 60,
                                 width: 850,
                                 decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.white.withOpacity(0.2)),
                                 ),
                                 child: Row(
                    children: [
                      const SizedBox(width: 16),
                      Expanded(
                        child: TextField(
                          controller: _linkController,
                          style: const TextStyle(color: Colors.white),
                          decoration: const InputDecoration(
                            border: InputBorder.none,
                            hintText: 'Paste the YouTube link',
                            hintStyle: TextStyle(color: Colors.white70),
                          ),
                        ),
                      ),
                      Container(
                        margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
                        child: ElevatedButton(
                          onPressed: _isLoading ? null : _handleGenerate,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color.fromARGB(255, 14, 114, 196),
                            padding: const EdgeInsets.symmetric(horizontal: 20),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                          child: _isLoading
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(
                                    color: Colors.white,
                                    strokeWidth: 2,
                                  ),
                                )
                              : const Text(
                                  'Generate',
                                  style: TextStyle(color: Colors.white),
                                ),
                        ),
                      ),
                    ],
                                 ),
                               ),)),
                 ),
               ),
               const SizedBox(height: 150,),
               Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: features.map((feature) {
          return Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8.0),
              child: GlassCard(
                icon: feature['icon'],
                title: feature['title'],
                description: feature['description'],
              ),
            ),
          );
        }).toList(),
      ),
    
            ],
          ),
        ),
      ),
    );
  }
}   
class GlassCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String description;

  const GlassCard({
    required this.icon,
    required this.title,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
           height: 180,
          padding: EdgeInsets.all(16),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [
                Colors.white.withOpacity(0.1),
                Colors.white.withOpacity(0.05),
              ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.white.withOpacity(0.2)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.15),
                blurRadius: 8,
                offset: Offset(0, 4),
              )
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, color: Colors.white, size: 32),
              SizedBox(height: 12),
              Text(
                title,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 8),
              Text(
                description,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white.withOpacity(0.9),
                  fontSize: 14,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}