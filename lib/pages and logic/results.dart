import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_markdown/flutter_markdown.dart';

class Message {
  final String text;
  final bool isUser;
  final List<Map<String, dynamic>>? sources;

  Message({
    required this.text,
    required this.isUser,
    this.sources,
  });
}

class ResultsPage extends StatefulWidget {
  final Map<String, dynamic> videoData;

  const ResultsPage({super.key, required this.videoData});

  @override
  State<ResultsPage> createState() => _ResultsPageState();
}

class _ResultsPageState extends State<ResultsPage> {
  final TextEditingController _chatController = TextEditingController();
  final List<Message> _chatMessages = [];
  bool _sendingMessage = false;
  String? _sessionId;
  String? _transcript;
  String? _summary;
  bool _isInitialLoading = true;

  @override
  void initState() {
    super.initState();
    print('Initializing ResultsPage with data: ${widget.videoData}');
    print('Available keys in videoData: ${widget.videoData.keys.toList()}');
    
    // Initialize data from the video processing response
    // First try transcript_id as that's what the backend uses
    _sessionId = widget.videoData['transcript_id']?.toString();
    
    if (_sessionId == null) {
      // Fallback to session_id if transcript_id is not available
      _sessionId = widget.videoData['session_id']?.toString();
    }
    
    print('Session ID from transcript_id: ${widget.videoData['transcript_id']}');
    print('Session ID from session_id: ${widget.videoData['session_id']}');
    print('Final session ID: $_sessionId');
    
    if (_sessionId == null) {
      print('ERROR: No session ID found in video data');
      print('Raw video data: ${widget.videoData}');
      print('Available keys: ${widget.videoData.keys.toList()}');
    } else {
      print('Successfully initialized with session ID: $_sessionId');
    }
    
    _transcript = widget.videoData['transcript']?.toString();
    _summary = widget.videoData['summary']?.toString();
    
    // If we have a session ID, we're not loading
    _isInitialLoading = _sessionId == null;

    // Add welcome message if we have a session
    if (_sessionId != null) {
      _chatMessages.add(Message(
        text: "I've analyzed the video. Ask me anything about its content!",
        isUser: false,
        sources: null,
      ));
    }
  }

  @override
  Widget build(BuildContext context) {
    final data = widget.videoData;
    final bool isLoading = data['isLoading'] as bool? ?? false;
    final String? youtubeUrl = data['youtube_url']?.toString();
    final String title = data['title']?.toString() ?? 'Video Title';
    final int duration = (data['duration'] as num?)?.toInt() ?? 0;
    final String? thumbnailUrl = data['thumbnail_url']?.toString();

    if (isLoading) {
      return _buildLoadingScreen(youtubeUrl);
    }

    return Scaffold(
      body: Container(
        height: double.infinity,
        width: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              const Color.fromARGB(255, 6, 48, 81),
              Colors.black
            ],
            begin: Alignment.centerLeft,
            end: Alignment.centerRight
          )
        ),
        child: Row(
          children: [
            // Left side - Summary
            Expanded(
              child: _buildSummarySection(
                title: title,
                duration: duration,
                thumbnailUrl: thumbnailUrl,
                summary: _summary ?? 'Loading summary...',
              ),
            ),
            // Divider
            Container(
              width: 1,
              color: Colors.white.withOpacity(0.2),
            ),
            // Right side - Chat
            Expanded(
              child: _buildChatSection(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingScreen(String? youtubeUrl) {
    return Scaffold(
      body: Container(
        height: double.infinity,
        width: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              const Color.fromARGB(255, 6, 48, 81),
              Colors.black
            ],
            begin: Alignment.centerLeft,
            end: Alignment.centerRight
          )
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(
                color: Colors.white,
              ),
              const SizedBox(height: 24),
              Text(
                "Processing your video...",
                style: GoogleFonts.poppins(
                  fontSize: 18,
                  color: Colors.white,
                ),
              ),
              if (youtubeUrl != null) ...[
                const SizedBox(height: 16),
                Text(
                  youtubeUrl,
                  style: GoogleFonts.poppins(
                    fontSize: 14,
                    color: Colors.white70,
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSummarySection({
    required String title,
    required int duration,
    String? thumbnailUrl,
    required String summary,
  }) {
    return Container(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              IconButton(
                icon: const Icon(Icons.arrow_back, color: Colors.white),
                onPressed: () => Navigator.pop(context),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Text(
                  title,
                  style: GoogleFonts.poppins(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          // Thumbnail
          if (thumbnailUrl != null)
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.network(
                thumbnailUrl,
                height: 200,
                width: double.infinity,
                fit: BoxFit.cover,
                errorBuilder: (context, error, stackTrace) {
                  return Container(
                    height: 200,
                    width: double.infinity,
                    color: Colors.grey[800],
                    child: const Icon(
                      Icons.video_library,
                      color: Colors.white54,
                      size: 48,
                    ),
                  );
                },
              ),
            ),
          const SizedBox(height: 24),
          // Summary with Markdown
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                child: Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.white.withOpacity(0.2)),
                  ),
                  child: Markdown(
                    data: summary,
                    styleSheet: MarkdownStyleSheet(
                      p: GoogleFonts.poppins(
                        fontSize: 16,
                        color: Colors.white,
                        height: 1.6,
                      ),
                      h1: GoogleFonts.poppins(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                      h2: GoogleFonts.poppins(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                      h3: GoogleFonts.poppins(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                      strong: GoogleFonts.poppins(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                      em: GoogleFonts.poppins(
                        fontSize: 16,
                        fontStyle: FontStyle.italic,
                        color: Colors.white,
                      ),
                      blockquote: GoogleFonts.poppins(
                        fontSize: 16,
                        color: Colors.white70,
                        fontStyle: FontStyle.italic,
                      ),
                      code: GoogleFonts.poppins(
                        fontSize: 16,
                        color: Colors.white,
                        backgroundColor: Colors.white.withOpacity(0.1),
                      ),
                      codeblockDecoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    selectable: true,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChatSection() {
    print('Building chat section with session ID: $_sessionId');
    print('Current video data: ${widget.videoData}');
    
    return Container(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          // Chat messages
          Expanded(
            child: _chatMessages.isEmpty
                ? Center(
                    child: Text(
                      _sessionId == null
                          ? "Process a video first to chat about it"
                          : "Ask me anything about the video!",
                      style: TextStyle(color: Colors.white70),
                    ),
                  )
                : ListView.builder(
                    padding: EdgeInsets.all(10),
                    itemCount: _chatMessages.length,
                    itemBuilder: (context, index) {
                      final message = _chatMessages[index];
                      return _buildMessageBubble(message);
                    },
                  ),
          ),
          
          // Show typing indicator when waiting for response
          if (_sendingMessage)
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Row(
                children: [
                  SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      color: Colors.white,
                      strokeWidth: 2,
                    ),
                  ),
                  SizedBox(width: 8),
                  Text(
                    "AI is thinking...",
                    style: TextStyle(
                      fontStyle: FontStyle.italic,
                      color: Colors.white70,
                    ),
                  ),
                ],
              ),
            ),
          
          // Chat input
          Container(
            margin: const EdgeInsets.only(top: 16),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.white.withOpacity(0.2)),
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _chatController,
                    style: const TextStyle(color: Colors.white),
                    decoration: const InputDecoration(
                      hintText: 'Ask anything about the video...',
                      hintStyle: TextStyle(color: Colors.white70),
                      border: InputBorder.none,
                      contentPadding: EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 12,
                      ),
                    ),
                    maxLines: null,
                    keyboardType: TextInputType.multiline,
                    enabled: !_sendingMessage,
                    onSubmitted: (_) => sendChatMessage(),
                  ),
                ),
                Container(
                  margin: const EdgeInsets.only(right: 8),
                  child: IconButton(
                    onPressed: _sendingMessage ? null : sendChatMessage,
                    icon: _sendingMessage
                        ? const SizedBox(
                            width: 24,
                            height: 24,
                            child: CircularProgressIndicator(
                              color: Colors.white,
                              strokeWidth: 2,
                            ),
                          )
                        : const Icon(Icons.send, color: Colors.white),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMessageBubble(Message message) {
    final bgColor = message.isUser 
        ? Colors.blue.withOpacity(0.2)
        : Colors.white.withOpacity(0.05);
    
    final alignment = message.isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start;
    final borderRadius = message.isUser
        ? BorderRadius.only(
            topLeft: Radius.circular(15),
            topRight: Radius.circular(15),
            bottomLeft: Radius.circular(15),
          )
        : BorderRadius.only(
            topLeft: Radius.circular(15),
            topRight: Radius.circular(15),
            bottomRight: Radius.circular(15),
          );
        
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Column(
        crossAxisAlignment: alignment,
        children: [
          Container(
            constraints: BoxConstraints(
              maxWidth: MediaQuery.of(context).size.width * 0.75,
            ),
            padding: EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: bgColor,
              borderRadius: borderRadius,
              border: Border.all(
                color: Colors.white.withOpacity(0.2),
                width: 1,
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  message.isUser ? "You" : "AI Assistant",
                  style: GoogleFonts.poppins(
                    fontWeight: FontWeight.bold,
                    fontSize: 12,
                    color: Colors.white70,
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  message.text,
                  style: GoogleFonts.poppins(
                    fontSize: 15,
                    color: Colors.white,
                  ),
                ),
                
                // Show sources if available
                if (!message.isUser && message.sources != null && message.sources!.isNotEmpty)
                  _buildSourcesSection(message.sources!),
              ],
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildSourcesSection(List<Map<String, dynamic>> sources) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(height: 8),
        Divider(height: 1, thickness: 1, color: Colors.white.withOpacity(0.3)),
        SizedBox(height: 4),
        Text(
          "Sources:",
          style: GoogleFonts.poppins(
            fontWeight: FontWeight.bold,
            fontSize: 12,
            color: Colors.white70,
          ),
        ),
        SizedBox(height: 4),
        ...sources.take(3).map((source) {
          final startTime = source['start_time'] is num 
              ? _formatTimestamp(source['start_time']) 
              : "Unknown";
              
          return Padding(
            padding: const EdgeInsets.only(top: 4.0),
            child: RichText(
              text: TextSpan(
                style: GoogleFonts.poppins(
                  fontSize: 12,
                  color: Colors.white70,
                ),
                children: [
                  TextSpan(
                    text: "• $startTime: ",
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  TextSpan(
                    text: _truncateText(source['text'] ?? "", 100),
                  ),
                ],
              ),
            ),
          );
        }).toList(),
        
        // Show more sources indicator if needed
        if (sources.length > 3)
          Padding(
            padding: const EdgeInsets.only(top: 4.0),
            child: Text(
              "+ ${sources.length - 3} more sources",
              style: GoogleFonts.poppins(
                fontSize: 12, 
                fontStyle: FontStyle.italic,
                color: Colors.white70,
              ),
            ),
          ),
      ],
    );
  }
  
  String _formatTimestamp(num seconds) {
    final mins = (seconds / 60).floor();
    final secs = (seconds % 60).floor();
    return "$mins:${secs.toString().padLeft(2, '0')}";
  }
  
  String _truncateText(String text, int maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + "...";
  }

  Future<void> sendChatMessage() async {
    if (_chatController.text.isEmpty) {
      return;
    }

    if (_sessionId == null) {
      print('Error: No session ID available for chat');
      print('Current video data: ${widget.videoData}');
      print('Available keys: ${widget.videoData.keys.toList()}');
      setState(() {
        _chatMessages.add(Message(
          text: "Error: No active session found. Please process the video again.",
          isUser: false,
        ));
      });
      return;
    }

    final message = _chatController.text;
    _chatController.clear();

    setState(() {
      _chatMessages.add(Message(text: message, isUser: true));
      _sendingMessage = true;
    });

    final url = Uri.parse("http://127.0.0.1:8000/chat");
    final body = jsonEncode({
      "query": message,
      "session_id": _sessionId,
    });

    try {
      print('Sending chat message to $url');
      print('Using session ID: $_sessionId');
      print('Request body: $body');
      
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: body,
      );

      print('Response status: ${response.statusCode}');
      print('Response body: ${response.body}');

      if (response.statusCode == 200) {
        try {
          final data = jsonDecode(response.body);
          print('Decoded chat response: $data');
          
          setState(() {
            _sendingMessage = false;
            
            if (data.containsKey('error')) {
              _chatMessages.add(Message(
                text: "Error: ${data['error']}",
                isUser: false,
              ));
            } else {
              _chatMessages.add(Message(
                text: data['answer'],
                isUser: false,
                sources: List<Map<String, dynamic>>.from(data['sources'] ?? []),
              ));
            }
          });
        } catch (jsonError) {
          print('Error parsing chat response: $jsonError');
          setState(() {
            _sendingMessage = false;
            _chatMessages.add(Message(
              text: "Sorry, I couldn't process that response.",
              isUser: false,
            ));
          });
        }
      } else {
        setState(() {
          _sendingMessage = false;
          _chatMessages.add(Message(
            text: "Error: Server responded with status ${response.statusCode}",
            isUser: false,
          ));
        });
      }
    } catch (e) {
      print('Exception during chat request: $e');
      setState(() {
        _sendingMessage = false;
        _chatMessages.add(Message(
          text: "Error: Couldn't connect to the server.",
          isUser: false,
        ));
      });
    }
  }
}