<!DOCTYPE html> 
<html lang="en"> 
<head> <!-- Basic meta stuff first. I always forget viewport if I don't put it here --> 
     <meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0">


<!-- Using CDN Tailwind for speed. Might migrate to local build later -->


<!-- Font Awesome icons (only using a few, but easier to include all) -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<!-- ================= HERO / INTRO ================= -->
<header class="py-24 px-6 text-center bg-gradient-to-br from-indigo-900 via-slate-900 to-black">
    
 <!-- Main headline -->
  <h1 class="text-6xl font-extrabold mb-4 tracking-tight">
         Engineering Portfolio
    </h1>

    <!-- Short intro text -->
   <p class="text-xl text-slate-400 max-w-3xl mx-auto mb-8">
        Specializing in Deep Learning frameworks from scratch and Autonomous AI Agents.
    </p>

    <!-- Tech stack badges (kept minimal on purpose) -->
   <div class="flex justify-center gap-4">
        <span class="bg-slate-800 border border-slate-700 px-4 py-2 rounded-lg text-sm font-mono">
            Python
        </span>
        <span class="bg-slate-800 border border-slate-700 px-4 py-2 rounded-lg text-sm font-mono">
            NumPy
        </span>
        <span class="bg-slate-800 border border-slate-700 px-4 py-2 rounded-lg text-sm font-mono">
            React
        </span>
    </div>
</header>


<!-- ================= PROJECTS SECTION ================= -->
<main class="max-w-6xl mx-auto py-20 px-6 grid grid-cols-1 md:grid-cols-2 gap-10">

    <!-- Project Card 1 -->
  <div class="group bg-slate-800/50 rounded-3xl p-8 border border-slate-700 hover:border-indigo-500 transition-all">
        
   <div class="text-indigo-400 mb-4">
            <i class="fas fa-microchip text-4xl"></i>
        </div>

   <h3 class="text-2xl font-bold mb-3">
            NeuralEngine.py
        </h3>

   <p class="text-slate-400 mb-6 font-light leading-relaxed">
            A custom Deep Learning framework built with pure NumPy. 
            Includes AdamW, L2 Regularization, and a flexible layer system.
            <!-- Might add benchmarks here later -->
        </p>

   <div class="flex items-center gap-4">
            <a href="https://github.com/YOUR_USER/REPO/blob/main/Neural_Engine/neuralnet.py"
               class="text-indigo-400 font-medium hover:text-white transition">
                View Code
            </a>

   <a href="#demo-link"
               class="bg-indigo-600 px-5 py-2 rounded-full text-sm font-semibold hover:bg-indigo-500">
                Run Demo
            </a>
        </div>
    </div>


    <!-- Project Card 2 -->
  <div class="group bg-slate-800/50 rounded-3xl p-8 border border-slate-700 hover:border-emerald-500 transition-all">
        
   <div class="text-emerald-400 mb-4">
            <i class="fas fa-robot text-4xl"></i>
        </div>

   <h3 class="text-2xl font-bold mb-3">
            Agentic Search AI
        </h3>

   <p class="text-slate-400 mb-6 font-light leading-relaxed">
            A Streamlit-based AI agent that autonomously scrapes the web,
            parses PDFs, and answers complex questions.
        </p>

   <div class="flex items-center gap-4">
            <a href="https://github.com/YOUR_USER/REPO/blob/main/AI_Agent/app.py"
               class="text-emerald-400 font-medium hover:text-white transition">
                View Code
            </a>

   <a href="https://streamlit.io/cloud"
               class="bg-emerald-600 px-5 py-2 rounded-full text-sm font-semibold hover:bg-emerald-500">
                Launch App
            </a>
        </div>
    </div>


    <!-- Project Card 3 -->
  <div class="group bg-slate-800/50 rounded-3xl p-8 border border-slate-700 hover:border-orange-500 transition-all">
        
  <div class="text-orange-400 mb-4">
            <i class="fas fa-calculator text-4xl"></i>
        </div>

   <h3 class="text-2xl font-bold mb-3">
            React Glassmorphism UI
        </h3>

  <p class="text-slate-400 mb-6 font-light leading-relaxed">
            A modern calculator UI focused on glassmorphism.
            Includes keyboard support and state-managed history.
        </p>
   <a href="./Calculator/index.html"
           class="inline-block text-orange-400 font-medium hover:text-white transition">
            Live Preview →
       </a>
  </div>


    <!-- Project Card 4 -->
   <div class="group bg-slate-800/50 rounded-3xl p-8 border border-slate-700 hover:border-cyan-500 transition-all">
        
  <div class="text-cyan-400 mb-4">
            <i class="fas fa-paper-plane text-4xl"></i>
        </div>
      <h3 class="text-2xl font-bold mb-3">
            Flight Productivity
        </h3>

   <p class="text-slate-400 mb-6 font-light leading-relaxed">
            A multi-threaded Python desktop app using the Pomodoro technique
            to maximize focused deep work sessions.
        </p>

  <a href="https://github.com/YOUR_USER/REPO/blob/main/Productivity/flightproductivity.py"
           class="inline-block text-cyan-400 font-medium hover:text-white transition">
            View Repo →
        </a>
  </div>

</main>


<!-- ================= FOOTER ================= -->
<footer class="py-12 text-center text-slate-600 border-t border-slate-800">
    <p>
        Built with GitHub Pages & a lot of late nights | 2026
    </p>
</footer>

</html>
