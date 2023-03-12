
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%  ___           _                 _   _      _                _ _          %
% |   \ _  _ _ _| |_  __ _ _ __   | | | |_ _ (___ _____ _ _ __(_| |_ _  _   %
% | |) | || | '_| ' \/ _` | '  \  | |_| | ' \| \ V / -_| '_(_-| |  _| || |  %
% |___/ \_,_|_| |_||_\__,_|_|_|_|  \___/|_||_|_|\_/\___|_| /__|_|\__|\_, |  %
%                                                                    |__/   %
% _________________________________________________________________________ %
%                                                                           %
%                                                                           %
% This is the duthesis class that gives a format of the thesis              %          
% according to the university of Durham regulation.  To use it, include     %
% the following in your preamble:                                           %
% \documentclass{duthesis}                                                  %
%                                                                           %
% This class file requires thesis.sty, which must be placed under the       %
% library directory.                                                        %
%                                                                           %
%                       Modified by Samet Akcay, 2019                       %
%                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% INIT CLASS FILE                                                           %
% =======                                                                   %
\NeedsTeXFormat{LaTeX2e}                                                    %
\ProvidesClass{duthesis}[2001/06/18 Durham Maths thesis class]              %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% PT SIZE                                                                   %
% =======                                                                   % 
\newcommand\@duthesisptsize{12pt}                                           %
\DeclareOption{12pt}{\renewcommand\@duthesisptsize{12pt}}                   %
\DeclareOption{11pt}{\renewcommand\@duthesisptsize{11pt}}                   %
\DeclareOption{10pt}{\renewcommand\@duthesisptsize{10pt}}                   %
\ProcessOptions                                                             %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% LOAD CLASS                                                                %
% =======                                                                   % 
\LoadClass[\@duthesisptsize,a4paper]{report}                                %
\pagestyle{plain}                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% PACKAGES                                                                  %
% ========                                                                  % 
\RequirePackage{lipsum}                                                     %
\RequirePackage{setspace}                                                   %
\RequirePackage{library/thesis}                                             %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% PAGE MARGINS                                                              %
% ============                                                              %
\setlength{\voffset}{-1in}                                                  %
\setlength{\marginparsep}{0in}                                              %
\setlength{\marginparwidth}{0in}                                            %
\setlength{\marginparpush}{0in}                                             %
\setlength{\hoffset}{-1in}                                                  %
\setlength{\oddsidemargin}{3.8cm}                                           %
\setlength{\evensidemargin}{0in}                                            %
\setlength{\textwidth}{15cm}                                                %
\setlength{\topmargin}{2cm}                                                 %
\setlength{\headheight}{15pt}                                               %
\setlength{\headsep}{16pt}                                                  %
\setlength{\textheight}{23.8cm}                                             %
\setlength{\footskip}{18pt}                                                 %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% INPUT VARIABLES (Name, Title, etc.)                                       %
% ===================================                                       %
%                                                                           %
% Store the title of the thesis.                                            %
\newcommand{\du@title}{Theory of Everything}                                %
\renewcommand{\title}[1]{\renewcommand{\du@title}{#1}}                      %
%                                                                           %
% Store the author's name.                                                  %
\newcommand{\du@author}{Richard Feynman}                                    %
\renewcommand{\author}[1]{\renewcommand{\du@author}{#1}}                    %
%                                                                           %
% Store the department name.                                                %
\newcommand{\du@dept}{ESM}                                                  %
\providecommand{\dept}[1]{\renewcommand{\du@dept}{#1}}                      %
%                                                                           %        
% Store the  degree.                                                        %
\newcommand{\du@degree}{Doctor of Philosophy}                               %
\providecommand{\degree}[1]{\renewcommand{\du@degree}{#1}}                  %
%                                                                           %
% Store the date the degree will be conferred.                              %
\newcommand{\du@degreedate}{May 1900}                                       %
\providecommand{\degreedate}[1]{\renewcommand{\du@degreedate}{#1}}          %
%                                                                           %
% Store the year of the copyright.                                          %
\newcommand{\du@copyrightyear}{1900}                                        %
\providecommand{\copyrightyear}[1]{\renewcommand{\du@copyrightyear}{#1}}    %    
%                                                                           %
% Store the document type.                                                  %
\newcommand{\du@documenttype}{Thesis}                                       %
\providecommand{\documenttype}[1]{\renewcommand{\du@documenttype}{#1}}      %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% TITLE                                                                     %
% =====                                                                     %
\renewcommand{\titlepage}{                                                  %
    \begin{center}                                                          %
        % TITLE                                                             %
        \vspace*{1cm}                                                       %
        {\Huge \bf \du@title\\}                                               %
                                                                            %          
        % AUTHOR                                                            %
        \vspace*{2cm}                                                       %
        {\LARGE\bf \du@author}                                              %
                                                                            %
        \vfill                                                              %                                                        %
        \vspace*{0.9cm}                                                     %
                                                                            %
        % Put your university logo here if you wish.                        % 
        \begin{center}                                                      %
            \includegraphics[width=6cm]{library/logo/durham-logo.png}                           %
        \end{center}                                                        %
                                                                            %
        \begin{singlespace}                                                 %
            \du@dept            \\                                          %
            Durham University   \\                                          %
            United Kingdom      \\                                          %
            \du@degreedate                                                  %
        \end{singlespace}                                                   %
                                                                            %
    \end{center}                                                            %
}                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% ABSTRACT                                                                  %
% ========                                                                  %
\renewcommand{\abstract}[1]{%                                               %
    \pagestyle{plain}                                                       %
    \chapter*{Abstract}                                                     %
    \addcontentsline{toc}{chapter}{\numberline{}Abstract}                   %
        \begin{singlespace}                                                 %
            \input{#1}                                                      %
        \end{singlespace}                                                   %
    \newpage                                                                %
}                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% DEDICATION                                                                %
% ==========                                                                %
\newcommand{\dedication}[2]{                                                %
    \chapter*{#2}                                                           %
    \addcontentsline{toc}{chapter}{\numberline{}#2}                         %
    \begin{singlespace}                                                     %
        \input{#1}                                                          %
    \end{singlespace}                                                       %
}                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% DECLARATION                                                               %
% ===========                                                               %
\newcommand{\declaration}{                                                  %
    \chapter*{Declaration}                                                  %
    \addcontentsline{toc}{chapter}{\numberline{}Declaration}                %
    This piece of work is a result of my own work except where it forms an assessment based on group project work. In the case of a group project, the work has been prepared in collaboration with other members of the group. Material from the work of others not involved in the project has been acknowledged and quotations and paraphrases suitably indicated.
    \hspace*{\fill}             %
}                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% ACKNOWLEDGEMENT                                                           %
% ===============                                                           %
\newcommand{\acknowledgement}[1]{                                           %
    \chapter*{Acknowledgements}                                             %
    \addcontentsline{toc}{chapter}{\numberline{}Acknowledgements}           % 
    \begin{singlespace}                                                     %
        \input{#1}                                                          %    
    \end{singlespace}                                                       %
}                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% TABLE OF CONTENTS                                                         %
% =================                                                         %
\makeatletter                                                               %
\renewcommand\tableofcontents{%                                             %
    \if@twocolumn                                                           %
      \@restonecoltrue\onecolumn                                            %
    \else                                                                   %
      \@restonecolfalse                                                     %
    \fi                                                                     %
    \chapter*{\contentsname                                                 %
        \@mkboth{\contentsname}{\contentsname}}%                            %
    \@starttoc{toc}%                                                        %
    \if@restonecol\twocolumn\fi                                             %
}                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% LIST OF FIGURES                                                           %
% ===============                                                           %
\renewcommand\listoffigures{%                                               %
    \if@twocolumn                                                           %
      \@restonecoltrue\onecolumn                                            %
    \else                                                                   %
      \@restonecolfalse                                                     %
    \fi                                                                     %
    \chapter*{\listfigurename                                               %
      \@mkboth{\listfigurename}{\listfigurename}}%                          %
    \addcontentsline{toc}{chapter}{\numberline{}List of Figures}            %
    \@starttoc{lof}%                                                        %
    \if@restonecol\twocolumn\fi                                             %
}                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% LIST OF TABLES                                                            %
% ==============                                                            %
\renewcommand\listoftables{%                                                %
    \if@twocolumn                                                           %
      \@restonecoltrue\onecolumn                                            %
    \else                                                                   %
      \@restonecolfalse                                                     %
    \fi                                                                     %
    \chapter*{\listtablename                                                %
      \@mkboth{\listtablename}{\listtablename}}%                            %
    \addcontentsline{toc}{chapter}{\numberline{}List of Tables}             %
    \@starttoc{lot}%                                                        %
    \if@restonecol\twocolumn\fi                                             %
}                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% LIST OF SYMBOLS                                                           %   
% ===============                                                           %
\newcommand{\listofsymbols}[1]{                                             %
    \chapter*{List of Symbols}                                              %
    \addcontentsline{toc}{chapter}{\numberline{}List of Symbols}            %
    \begin{singlespace}                                                     %
        \input{#1}                                                          %
    \end{singlespace}                                                       %
}                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% FRONT MATTER                                                              %
% ============                                                              %
\DeclareRobustCommand{\frontmatter}{                                        %
    \pagenumbering{roman}                                                   %
    \setcounter{page}{1}                                                    %
    \newpage                                                                %
    \thispagestyle{empty}                                                   %
}                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% MAIN MATTER                                                               %
% ===========                                                               %
\DeclareRobustCommand{\mainmatter}{                                         %
    \clearpage                                                              %
    \pagenumbering{arabic}                                                  %
    \setcounter{page}{1}                                                    %
}                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% THEBIBLIOGRAPHY                                                           %
% ===============                                                           %   
\renewenvironment{thebibliography}[1]{                                      %  
    \chapter*{\bibname                                                      %
        \@mkboth{\bibname}{\bibname}}%                                      %
      \list{\@biblabel{\@arabic\c@enumiv}}%                                 %
           {\settowidth\labelwidth{\@biblabel{#1}}%                         %
            \leftmargin\labelwidth                                          %
            \advance\leftmargin\labelsep                                    %
            \@openbib@code                                                  %
            \usecounter{enumiv}%                                            %
            \let\p@enumiv\@empty                                            %
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%                    %
      \sloppy                                                               %
      \clubpenalty4000                                                      %
      \@clubpenalty \clubpenalty                                            %
      \widowpenalty4000%                                                    %
      \sfcode`\.\@m                                                         %
}                                                                           %
     {\def\@noitemerr                                                       %
       {\@latex@warning{Empty `thebibliography' environment}}%              %
      \endlist}                                                             %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% APPENDIX                                                                  %
% ========                                                                  %
\renewcommand{\appendix}{                                                   %    
    \setcounter{chapter}{0}                                                 %
    \renewcommand{\chaptername}{Appendix}                                   %
    \renewcommand{\thechapter}{\Alph{chapter}}                              %
    \renewcommand{\theequation}                                             %
                 {\Alph{chapter}.\arabic{section}.\arabic{equation}}        %
    \addcontentsline{toc}{chapter}{\numberline{}Appendix}                   %
    \setcounter{equation}{0}                                                %
}                                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
