\version "2.24.0"

\header {
  title = "Test"
  subtitle = "Drum Roadmap (CCM Auto-Transcribed)"
  composer = ""
  tagline = ""
}

\paper {
  #(set-paper-size "a4")
  system-system-spacing.basic-distance = #15
  markup-system-spacing.basic-distance = #5
  top-margin = 15
  bottom-margin = 15
  left-margin = 20
  right-margin = 20
}

\score {
  \new DrumStaff \with {
    drumStyleTable = #percussion-style
    \numericTimeSignature
  } {
    \drummode {
      \tempo 4=120
      \time 4/4
  \mark \markup { \bold "Intro" }
  << { \stemUp r16 r16 cymca16 r16 sn16 r16 cymca16 r16 sn16 r16 cymca16 r16 sn16 r16 cymca16 r16 } \\ { \stemDown r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 } >>
  \mark \markup { \bold "Groove" }
  << { \stemUp sn16 r16 cymca16 r16 sn16 r16 cymca16 r16 sn16 r16 cymca16 r16 sn16 r16 cymca16 r16 } \\ { \stemDown r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 } >>
  \mark \markup { \bold "Fill" }
  << { \stemUp sn16 r16 cymr16 r16 sn16 r16 cymca16 r16 sn16 r16 cymca16 r16 sn16 r16 cymca16 r16 } \\ { \stemDown r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 } >>
  \mark \markup { \bold "Outro" }
  \mark \markup { \italic \small "Fill" }
  << { \stemUp r16 r16 r16 r16 r16 r16 r16 tomfl16 r16 r16 r16 tomfl16 sn16 r16 r16 tomfl16 } \\ { \stemDown bd16 bd16 bd16 r16 bd16 bd16 bd16 r16 bd16 r16 bd16 r16 r16 r16 bd16 r16 } >>
      \break
  << { \stemUp r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 } \\ { \stemDown r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 r16 } >>
    }}
  }}
  \layout {{
    \context {{
      \DrumStaff
      \consists "Merge_rests_engraver"
    }}
  }}
}}
