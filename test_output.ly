\version "2.24.0"
\header {
  title = "Test"
  subtitle = "CCM Drum Roadmap — PAS Standard  (BPM: 60)"
  composer = "Test"
  tagline = ""
}
\paper {
  #(set-paper-size "a4")
  system-system-spacing.basic-distance = #18
  score-markup-spacing.basic-distance = #3
  markup-system-spacing.basic-distance = #5
  top-margin = #12
  bottom-margin = #12
  left-margin = #15
  right-margin = #15
  print-page-number = ##t
  max-systems-per-page = 4
}

\score {
  \new DrumStaff \with {
    drumStyleTable = #percussion-style
    \numericTimeSignature
    fontSize = #-1
    \override StaffSymbol.staff-space = #(magstep -1)
  } <<
    \new DrumVoice {
      \voiceOne
      \drummode {
        \tempo 4 = 60
        \time 4/4
        \mark \markup { \bold \box \large "Intro" }
    << { \stemUp r4 sn4 r4 sn4 } \\ { \stemDown bd4 r4 bd4 r4 } >>
        \mark \markup { \italic \small "Fill" }
    << { \stemUp r4 r4 r4 r4 } \\ { \stemDown r4 r4 r4 r4 } >>
      }
    }
  >>
  \layout {
    indent = 2\cm
    \context {
      \DrumStaff
      \consists "Merge_rests_engraver"
      \override BarLine.hair-thickness = #3.5
    }
  }
}
\markup {
  \vspace #0.5
  \fill-line {
    \rounded-box \pad-markup #1.5 \column {
      \line { \bold "[구간 리듬]  " "8비트 클로즈드 하이햇 기본 그루브" }
      \line { \bold "[줄별 필인]  " "해당 없음" }
    }
  }
}
\markup { \vspace #1.5 }
