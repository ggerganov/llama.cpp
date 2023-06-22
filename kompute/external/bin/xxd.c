/*
As indicated at https://lists.debian.org/debian-legal/2015/01/msg00037.html,
the author has permitted redistribution of xxd under the MIT license, as follows:

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * xxd: my hexdump facility. jw
 *
 *  2.10.90 changed to word output
 *  3.03.93 new indent style, dumb bug inserted and fixed.
 *	    -c option, mls
 * 26.04.94 better option parser, -ps, -l, -s added.
 *  1.07.94 -r badly needs - as input file.  Per default autoskip over
 *	       consecutive lines of zeroes, as unix od does.
 *	    -a shows them too.
 *	    -i dump as c-style #include "file.h"
 *  1.11.95 if "xxd -i" knows the filename, an 'unsigned char filename_bits[]'
 *	    array is written in correct c-syntax.
 *	    -s improved, now defaults to absolute seek, relative requires a '+'.
 *	    -r improved, now -r -s -0x... is supported.
 *	       change/suppress leading '\0' bytes.
 *	    -l n improved: stops exactly after n bytes.
 *	    -r improved, better handling of partial lines with trailing garbage.
 *	    -r improved, now -r -p works again!
 *	    -r improved, less flushing, much faster now! (that was silly)
 *  3.04.96 Per repeated request of a single person: autoskip defaults to off.
 * 15.05.96 -v added. They want to know the version.
 *	    -a fixed, to show last line inf file ends in all zeros.
 *	    -u added: Print upper case hex-letters, as preferred by unix bc.
 *	    -h added to usage message. Usage message extended.
 *	    Now using outfile if specified even in normal mode, aehem.
 *	    No longer mixing of ints and longs. May help doze people.
 *	    Added binify ioctl for same reason. (Enough Doze stress for 1996!)
 * 16.05.96 -p improved, removed occasional superfluous linefeed.
 * 20.05.96 -l 0 fixed. tried to read anyway.
 * 21.05.96 -i fixed. now honours -u, and prepends __ to numeric filenames.
 *	    compile -DWIN32 for NT or W95. George V. Reilly, * -v improved :-)
 *	    support --gnuish-longhorn-options
 * 25.05.96 MAC support added: CodeWarrior already uses ``outline'' in Types.h
 *	    which is included by MacHeaders (Axel Kielhorn). Renamed to
 *	    xxdline().
 *  7.06.96 -i printed 'int' instead of 'char'. *blush*
 *	    added Bram's OS2 ifdefs...
 * 18.07.96 gcc -Wall @ SunOS4 is now slient.
 *	    Added osver for MSDOS/DJGPP/WIN32.
 * 29.08.96 Added size_t to strncmp() for Amiga.
 * 24.03.97 Windows NT support (Phil Hanna). Clean exit for Amiga WB (Bram)
 * 02.04.97 Added -E option, to have EBCDIC translation instead of ASCII
 *	    (azc10@yahoo.com)
 * 22.05.97 added -g (group octets) option (jcook@namerica.kla.com).
 * 23.09.98 nasty -p -r misfeature fixed: slightly wrong output, when -c was
 *	    missing or wrong.
 * 26.09.98 Fixed: 'xxd -i infile outfile' did not truncate outfile.
 * 27.10.98 Fixed: -g option parser required blank.
 *	    option -b added: 01000101 binary output in normal format.
 * 16.05.00 Added VAXC changes by Stephen P. Wall
 * 16.05.00 Improved MMS file and merge for VMS by Zoltan Arpadffy
 *
 * (c) 1990-1998 by Juergen Weigert (jnweiger@informatik.uni-erlangen.de)
 *
 * Small changes made afterwards by Bram Moolenaar et al.
 *
 * Distribute freely and credit me,
 * make money and share with me,
 * lose money and don't ask me.
 *
 *
 */

/* Visual Studio 2005 has 'deprecated' many of the standard CRT functions */
#if _MSC_VER >= 1400
# define _CRT_SECURE_NO_DEPRECATE
# define _CRT_NONSTDC_NO_DEPRECATE
#endif

#include <stdio.h>
#ifdef VAXC
# include <file.h>
#else
# include <fcntl.h>
#endif
#ifdef __TSC__
# define MSDOS
#endif
#if !defined(OS2) && defined(__EMX__)
# define OS2
#endif
#if defined(MSDOS) || defined(WIN32) || defined(OS2) || defined(__BORLANDC__) || defined(CYGWIN)
# include <io.h>	/* for setmode() */
#else
# ifdef UNIX
#  include <unistd.h>
# endif
#endif
#include <stdlib.h>
#include <string.h>	/* for strncmp() */
#include <ctype.h>	/* for isalnum() */
#if __MWERKS__ && !defined(BEBOX)
# include <unix.h>	/* for fdopen() on MAC */
#endif

#if defined(__BORLANDC__) && __BORLANDC__ <= 0x0410 && !defined(fileno)
/* Missing define and prototype grabbed from the BC 4.0 <stdio.h> */
# define fileno(f)       ((f)->fd)
FILE   _FAR *_Cdecl _FARFUNC fdopen(int __handle, char _FAR *__type);
#endif


/*  This corrects the problem of missing prototypes for certain functions
 *  in some GNU installations (e.g. SunOS 4.1.x).
 *  Darren Hiebert <darren@hmi.com> (sparc-sun-sunos4.1.3_U1/2.7.2.2)
 */
#if defined(__GNUC__) && defined(__STDC__)
# ifndef __USE_FIXED_PROTOTYPES__
#  define __USE_FIXED_PROTOTYPES__
# endif
#endif

#ifndef __USE_FIXED_PROTOTYPES__
/*
 * This is historic and works only if the compiler really has no prototypes:
 *
 * Include prototypes for Sun OS 4.x, when using an ANSI compiler.
 * FILE is defined on OS 4.x, not on 5.x (Solaris).
 * if __SVR4 is defined (some Solaris versions), don't include this.
 */
#if defined(sun) && defined(FILE) && !defined(__SVR4) && defined(__STDC__)
#  define __P(a) a
/* excerpt from my sun_stdlib.h */
extern int fprintf __P((FILE *, char *, ...));
extern int fputs   __P((char *, FILE *));
extern int _flsbuf __P((unsigned char, FILE *));
extern int _filbuf __P((FILE *));
extern int fflush  __P((FILE *));
extern int fclose  __P((FILE *));
extern int fseek   __P((FILE *, long, int));
extern int rewind  __P((FILE *));

extern void perror __P((char *));
# endif
#endif

extern long int strtol();
extern long int ftell();

char version[] = "xxd V1.10 27oct98 by Juergen Weigert";
#ifdef WIN32
char osver[] = " (Win32)";
#else
# ifdef DJGPP
char osver[] = " (dos 32 bit)";
# else
#  ifdef MSDOS
char osver[] = " (dos 16 bit)";
#  else
char osver[] = "";
#  endif
# endif
#endif

#if !defined(CYGWIN) && (defined(CYGWIN32) || defined(__CYGWIN__) || defined(__CYGWIN32__))
# define CYGWIN
#endif
#if defined(MSDOS) || defined(WIN32) || defined(OS2)
# define BIN_READ(yes)  ((yes) ? "rb" : "rt")
# define BIN_WRITE(yes) ((yes) ? "wb" : "wt")
# define BIN_CREAT(yes) ((yes) ? (O_CREAT|O_BINARY) : O_CREAT)
# define BIN_ASSIGN(fp, yes) setmode(fileno(fp), (yes) ? O_BINARY : O_TEXT)
# define PATH_SEP '\\'
#elif defined(CYGWIN)
# define BIN_READ(yes)  ((yes) ? "rb" : "rt")
# define BIN_WRITE(yes) ((yes) ? "wb" : "w")
# define BIN_CREAT(yes) ((yes) ? (O_CREAT|O_BINARY) : O_CREAT)
# define BIN_ASSIGN(fp, yes) ((yes) ? (void) setmode(fileno(fp), O_BINARY) : (void) (fp))
# define PATH_SEP '/'
#else
# ifdef VMS
#  define BIN_READ(dummy)  "r"
#  define BIN_WRITE(dummy) "w"
#  define BIN_CREAT(dummy) O_CREAT
#  define BIN_ASSIGN(fp, dummy) fp
#  define PATH_SEP ']'
#  define FILE_SEP '.'
# else
#  define BIN_READ(dummy)  "r"
#  define BIN_WRITE(dummy) "w"
#  define BIN_CREAT(dummy) O_CREAT
#  define BIN_ASSIGN(fp, dummy) fp
#  define PATH_SEP '/'
# endif
#endif

/* open has only to arguments on the Mac */
#if __MWERKS__
# define OPEN(name, mode, umask) open(name, mode)
#else
# define OPEN(name, mode, umask) open(name, mode, umask)
#endif

#ifdef AMIGA
# define STRNCMP(s1, s2, l) strncmp(s1, s2, (size_t)l)
#else
# define STRNCMP(s1, s2, l) strncmp(s1, s2, l)
#endif

#ifndef __P
# if defined(__STDC__) || defined(MSDOS) || defined(WIN32) || defined(OS2) \
        || defined(__BORLANDC__)
#  define __P(a) a
# else
#  define __P(a) ()
# endif
#endif

/* Let's collect some prototypes */
/* CodeWarrior is really picky about missing prototypes */
static void exit_with_usage __P((char *));
static int huntype __P((FILE *, FILE *, FILE *, char *, int, int, long));
static void xxdline __P((FILE *, char *, int));

#define TRY_SEEK	/* attempt to use lseek, or skip forward by reading */
#define COLS 256	/* change here, if you ever need more columns */
#define LLEN (11 + (9*COLS-1)/1 + COLS + 2)

char hexxa[] = "0123456789abcdef0123456789ABCDEF", *hexx = hexxa;

/* the different hextypes known by this program: */
#define HEX_NORMAL 0
#define HEX_POSTSCRIPT 1
#define HEX_CINCLUDE 2
#define HEX_BITS 3		/* not hex a dump, but bits: 01111001 */

static void
exit_with_usage(pname)
char *pname;
{
  fprintf(stderr, "Usage:\n       %s [options] [infile [outfile]]\n", pname);
  fprintf(stderr, "    or\n       %s -r [-s [-]offset] [-c cols] [-ps] [infile [outfile]]\n", pname);
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "    -a          toggle autoskip: A single '*' replaces nul-lines. Default off.\n");
  fprintf(stderr, "    -b          binary digit dump (incompatible with -p,-i,-r). Default hex.\n");
  fprintf(stderr, "    -c cols     format <cols> octets per line. Default 16 (-i: 12, -ps: 30).\n");
  fprintf(stderr, "    -E          show characters in EBCDIC. Default ASCII.\n");
  fprintf(stderr, "    -g          number of octets per group in normal output. Default 2.\n");
  fprintf(stderr, "    -h          print this summary.\n");
  fprintf(stderr, "    -i          output in C include file style.\n");
  fprintf(stderr, "    -l len      stop after <len> octets.\n");
  fprintf(stderr, "    -ps         output in postscript plain hexdump style.\n");
  fprintf(stderr, "    -r          reverse operation: convert (or patch) hexdump into binary.\n");
  fprintf(stderr, "    -r -s off   revert with <off> added to file positions found in hexdump.\n");
  fprintf(stderr, "    -s %sseek  start at <seek> bytes abs. %sinfile offset.\n",
#ifdef TRY_SEEK
      "[+][-]", "(or +: rel.) ");
#else
      "", "");
#endif
  fprintf(stderr, "    -u          use upper case hex letters.\n");
  fprintf(stderr, "    -v          show version: \"%s%s\".\n", version, osver);
  exit(1);
}

/*
 * Max. cols binary characters are decoded from the input stream per line.
 * Two adjacent garbage characters after evaluated data delimit valid data.
 * Everything up to the next newline is discarded.
 *
 * The name is historic and came from 'undo type opt h'.
 */
static int
huntype(fpi, fpo, fperr, pname, cols, hextype, base_off)
FILE *fpi, *fpo, *fperr;
char *pname;
int cols, hextype;
long base_off;
{
  int c, ign_garb = 1, n1 = -1, n2 = 0, n3, p = cols;
  long have_off = 0, want_off = 0;

  rewind(fpi);

  while ((c = getc(fpi)) != EOF)
    {
      if (c == '\r')	/* Doze style input file? */
    continue;

#if 0	/* this doesn't work when there is normal text after the hex codes in
       the last line that looks like hex */
      if (c == ' ' || c == '\n' || c == '\t')  /* allow multiple spaces */
    continue;
#endif

      n3 = n2;
      n2 = n1;

      if (c >= '0' && c <= '9')
    n1 = c - '0';
      else if (c >= 'a' && c <= 'f')
    n1 = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
    n1 = c - 'A' + 10;
      else
    {
      n1 = -1;
      if (ign_garb)
        continue;
    }

      ign_garb = 0;

      if (p >= cols)
    {
      if (!hextype)
        {
          if (n1 < 0)
        {
          p = 0;
          continue;
        }
          want_off = (want_off << 4) | n1;
          continue;
        }
      else
        p = 0;
    }

      if (base_off + want_off != have_off)
    {
      fflush(fpo);
#ifdef TRY_SEEK
      c = fseek(fpo, base_off + want_off - have_off, 1);
      if (c >= 0)
        have_off = base_off + want_off;
#endif
      if (base_off + want_off < have_off)
        {
          fprintf(fperr, "%s: sorry, cannot seek backwards.\n", pname);
          return 5;
        }
      for (; have_off < base_off + want_off; have_off++)
        putc(0, fpo);
    }

      if (n2 >= 0 && n1 >= 0)
    {
      putc((n2 << 4) | n1, fpo);
      have_off++;
      want_off++;
      n1 = -1;
      if ((++p >= cols) && !hextype)
        {
          /* skip rest of line as garbage */
          want_off = 0;
          while ((c = getc(fpi)) != '\n' && c != EOF)
        ;
          ign_garb = 1;
        }
    }
      else if (n1 < 0 && n2 < 0 && n3 < 0)
    {
      /* already stumbled into garbage, skip line, wait and see */
      if (!hextype)
        want_off = 0;
      while ((c = getc(fpi)) != '\n' && c != EOF)
        ;
      ign_garb = 1;
    }
    }
  fflush(fpo);
#ifdef TRY_SEEK
  fseek(fpo, 0L, 2);
#endif
  fclose(fpo);
  fclose(fpi);
  return 0;
}

/*
 * Print line l. If nz is false, xxdline regards the line a line of
 * zeroes. If there are three or more consecutive lines of zeroes,
 * they are replaced by a single '*' character.
 *
 * If the output ends with more than two lines of zeroes, you
 * should call xxdline again with l being the last line and nz
 * negative. This ensures that the last line is shown even when
 * it is all zeroes.
 *
 * If nz is always positive, lines are never suppressed.
 */
static void
xxdline(fp, l, nz)
FILE *fp;
char *l;
int nz;
{
  static char z[LLEN+1];
  static int zero_seen = 0;

  if (!nz && zero_seen == 1)
    strcpy(z, l);

  if (nz || !zero_seen++)
    {
      if (nz)
    {
      if (nz < 0)
        zero_seen--;
      if (zero_seen == 2)
        fputs(z, fp);
      if (zero_seen > 2)
        fputs("*\n", fp);
    }
      if (nz >= 0 || zero_seen > 0)
    fputs(l, fp);
      if (nz)
    zero_seen = 0;
    }
}

/* This is an EBCDIC to ASCII conversion table */
/* from a proposed BTL standard April 16, 1979 */
static unsigned char etoa64[] =
{
    0040,0240,0241,0242,0243,0244,0245,0246,
    0247,0250,0325,0056,0074,0050,0053,0174,
    0046,0251,0252,0253,0254,0255,0256,0257,
    0260,0261,0041,0044,0052,0051,0073,0176,
    0055,0057,0262,0263,0264,0265,0266,0267,
    0270,0271,0313,0054,0045,0137,0076,0077,
    0272,0273,0274,0275,0276,0277,0300,0301,
    0302,0140,0072,0043,0100,0047,0075,0042,
    0303,0141,0142,0143,0144,0145,0146,0147,
    0150,0151,0304,0305,0306,0307,0310,0311,
    0312,0152,0153,0154,0155,0156,0157,0160,
    0161,0162,0136,0314,0315,0316,0317,0320,
    0321,0345,0163,0164,0165,0166,0167,0170,
    0171,0172,0322,0323,0324,0133,0326,0327,
    0330,0331,0332,0333,0334,0335,0336,0337,
    0340,0341,0342,0343,0344,0135,0346,0347,
    0173,0101,0102,0103,0104,0105,0106,0107,
    0110,0111,0350,0351,0352,0353,0354,0355,
    0175,0112,0113,0114,0115,0116,0117,0120,
    0121,0122,0356,0357,0360,0361,0362,0363,
    0134,0237,0123,0124,0125,0126,0127,0130,
    0131,0132,0364,0365,0366,0367,0370,0371,
    0060,0061,0062,0063,0064,0065,0066,0067,
    0070,0071,0372,0373,0374,0375,0376,0377
};

const char* extract_filename(const char* path) {
    const char* filename = strrchr(path, '/');
    if (filename) {
        return filename + 1;
    }
    return path;
}

int
main(argc, argv)
int argc;
char *argv[];
{
  FILE *fp, *fpo;
  int c, e, p = 0, relseek = 1, negseek = 0, revert = 0;
  int cols = 0, nonzero = 0, autoskip = 0, hextype = HEX_NORMAL;
  int ebcdic = 0;
  int octspergrp = -1;	/* number of octets grouped in output */
  int grplen;		/* total chars per octet group */
  long length = -1, n = 0, seekoff = 0;
  char l[LLEN+1];
  char *pname, *pp;

#ifdef AMIGA
  /* This program doesn't work when started from the Workbench */
  if (argc == 0)
    exit(1);
#endif

  pname = argv[0];
  for (pp = pname; *pp; )
    if (*pp++ == PATH_SEP)
      pname = pp;
#ifdef FILE_SEP
  for (pp = pname; *pp; pp++)
    if (*pp == FILE_SEP)
      {
    *pp = '\0';
    break;
      }
#endif

  while (argc >= 2)
    {
      pp = argv[1] + (!STRNCMP(argv[1], "--", 2) && argv[1][2]);
       if (!STRNCMP(pp, "-a", 2)) autoskip = 1 - autoskip;
      else if (!STRNCMP(pp, "-b", 2)) hextype = HEX_BITS;
      else if (!STRNCMP(pp, "-u", 2)) hexx = hexxa + 16;
      else if (!STRNCMP(pp, "-p", 2)) hextype = HEX_POSTSCRIPT;
      else if (!STRNCMP(pp, "-i", 2)) hextype = HEX_CINCLUDE;
      else if (!STRNCMP(pp, "-r", 2)) revert++;
      else if (!STRNCMP(pp, "-E", 2)) ebcdic++;
      else if (!STRNCMP(pp, "-v", 2))
    {
      fprintf(stderr, "%s%s\n", version, osver);
      exit(0);
    }
      else if (!STRNCMP(pp, "-c", 2))
    {
      if (pp[2] && STRNCMP("ols", pp + 2, 3))
        cols = (int)strtol(pp + 2, NULL, 0);
      else
        {
          if (!argv[2])
        exit_with_usage(pname);
          cols = (int)strtol(argv[2], NULL, 0);
          argv++;
          argc--;
        }
    }
      else if (!STRNCMP(pp, "-g", 2))
    {
      if (pp[2] && STRNCMP("roupsize", pp + 2, 8))
        octspergrp = (int)strtol(pp + 2, NULL, 0);
      else
        {
          if (!argv[2])
        exit_with_usage(pname);
          octspergrp = (int)strtol(argv[2], NULL, 0);
          argv++;
          argc--;
        }
    }
      else if (!STRNCMP(pp, "-s", 2))
    {
      relseek = 0;
      negseek = 0;
      if (pp[2] && STRNCMP("kip", pp+2, 3) && STRNCMP("eek", pp+2, 3))
        {
#ifdef TRY_SEEK
          if (pp[2] == '+')
        relseek++;
          if (pp[2+relseek] == '-')
        negseek++;
#endif
          seekoff = strtol(pp + 2+relseek+negseek, (char **)NULL, 0);
        }
      else
        {
          if (!argv[2])
        exit_with_usage(pname);
#ifdef TRY_SEEK
          if (argv[2][0] == '+')
        relseek++;
          if (argv[2][relseek] == '-')
        negseek++;
#endif
          seekoff = strtol(argv[2] + relseek+negseek, (char **)NULL, 0);
          argv++;
          argc--;
        }
    }
      else if (!STRNCMP(pp, "-l", 2))
    {
      if (pp[2] && STRNCMP("en", pp + 2, 2))
        length = strtol(pp + 2, (char **)NULL, 0);
      else
        {
          if (!argv[2])
        exit_with_usage(pname);
          length = strtol(argv[2], (char **)NULL, 0);
          argv++;
          argc--;
        }
    }
      else if (!strcmp(pp, "--"))	/* end of options */
    {
      argv++;
      argc--;
      break;
    }
      else if (pp[0] == '-' && pp[1])	/* unknown option */
    exit_with_usage(pname);
      else
    break;				/* not an option */

      argv++;				/* advance to next argument */
      argc--;
    }

  if (!cols)
    switch (hextype)
      {
      case HEX_POSTSCRIPT:	cols = 30; break;
      case HEX_CINCLUDE:	cols = 12; break;
      case HEX_BITS:		cols = 6; break;
      case HEX_NORMAL:
      default:			cols = 16; break;
      }

  if (octspergrp < 0)
    switch (hextype)
      {
      case HEX_BITS:		octspergrp = 1; break;
      case HEX_NORMAL:		octspergrp = 2; break;
      case HEX_POSTSCRIPT:
      case HEX_CINCLUDE:
      default:			octspergrp = 0; break;
      }

  if (cols < 1 || ((hextype == HEX_NORMAL || hextype == HEX_BITS)
                                && (cols > COLS)))
    {
      fprintf(stderr, "%s: invalid number of columns (max. %d).\n", pname, COLS);
      exit(1);
    }

  if (octspergrp < 1)
    octspergrp = cols;

  if (argc > 3)
    exit_with_usage(pname);

  if (argc == 1 || (argv[1][0] == '-' && !argv[1][1]))
    BIN_ASSIGN(fp = stdin, !revert);
  else
    {
      if ((fp = fopen(argv[1], BIN_READ(!revert))) == NULL)
    {
      fprintf(stderr,"%s: ", pname);
      perror(argv[1]);
      return 2;
    }
    }

  if (argc < 3 || (argv[2][0] == '-' && !argv[2][1]))
    BIN_ASSIGN(fpo = stdout, revert);
  else
    {
      int fd;
      int mode = revert ? O_WRONLY : (O_TRUNC|O_WRONLY);

      if (((fd = OPEN(argv[2], mode | BIN_CREAT(revert), 0666)) < 0) ||
      (fpo = fdopen(fd, BIN_WRITE(revert))) == NULL)
    {
      fprintf(stderr, "%s: ", pname);
      perror(argv[2]);
      return 3;
    }
      rewind(fpo);
    }

  if (revert)
    {
      if (hextype && (hextype != HEX_POSTSCRIPT))
    {
      fprintf(stderr, "%s: sorry, cannot revert this type of hexdump\n", pname);
      return -1;
    }
      return huntype(fp, fpo, stderr, pname, cols, hextype,
        negseek ? -seekoff : seekoff);
    }

  if (seekoff || negseek || !relseek)
    {
#ifdef TRY_SEEK
      if (relseek)
    e = fseek(fp, negseek ? -seekoff : seekoff, 1);
      else
    e = fseek(fp, negseek ? -seekoff : seekoff, negseek ? 2 : 0);
      if (e < 0 && negseek)
    {
      fprintf(stderr, "%s: sorry cannot seek.\n", pname);
      return 4;
    }
      if (e >= 0)
    seekoff = ftell(fp);
      else
#endif
    {
      long s = seekoff;

      while (s--)
        (void)getc(fp);
    }
    }

  if (hextype == HEX_CINCLUDE)
    {
      const char* filename = extract_filename(argv[1]);

      if (fp != stdin)
    {
      fprintf(fpo, "unsigned char %s", isdigit((int)filename[0]) ? "__" : "");
      for (e = 0; (c = filename[e]) != 0; e++)
        putc(isalnum(c) ? c : '_', fpo);
      fputs("[] = {\n", fpo);
    }

      p = 0;
      while ((length < 0 || p < length) && (c = getc(fp)) != EOF)
    {
      fprintf(fpo, (hexx == hexxa) ? "%s0x%02x" : "%s0X%02X",
        (p % cols) ? ", " : ",\n  "+2*!p,  c);
      p++;
    }

      if (p)
    fputs("\n};\n"+3*(fp == stdin), fpo);

      if (fp != stdin)
    {
      fprintf(fpo, "unsigned int %s", isdigit((int)filename[0]) ? "__" : "");
      for (e = 0; (c = filename[e]) != 0; e++)
        putc(isalnum(c) ? c : '_', fpo);
      fprintf(fpo, "_len = %d;\n", p);
    }

      fclose(fp);
      fclose(fpo);
      return 0;
    }

  if (hextype == HEX_POSTSCRIPT)
    {
      p = cols;
      while ((length < 0 || n < length) && (e = getc(fp)) != EOF)
    {
      putchar(hexx[(e >> 4) & 0xf]);
      putchar(hexx[(e     ) & 0xf]);
      n++;
      if (!--p)
        {
          putchar('\n');
          p = cols;
        }
    }
      if (p < cols)
    putchar('\n');
      fclose(fp);
      fclose(fpo);
      return 0;
    }

  /* hextype: HEX_NORMAL or HEX_BITS */

  if (hextype == HEX_NORMAL)
    grplen = octspergrp + octspergrp + 1;	/* chars per octet group */
  else	/* hextype == HEX_BITS */
    grplen = 8 * octspergrp + 1;

  while ((length < 0 || n < length) && (e = getc(fp)) != EOF)
    {
      if (p == 0)
    {
      sprintf(l, "%07lx: ", n + seekoff);
      for (c = 9; c < LLEN; l[c++] = ' ');
    }
      if (hextype == HEX_NORMAL)
    {
      l[c = (9 + (grplen * p) / octspergrp)] = hexx[(e >> 4) & 0xf];
      l[++c]			       = hexx[ e       & 0xf];
    }
      else /* hextype == HEX_BITS */
    {
      int i;

      c = (9 + (grplen * p) / octspergrp) - 1;
      for (i = 7; i >= 0; i--)
        l[++c] = (e & (1 << i)) ? '1' : '0';
    }
      if (ebcdic)
    e = (e < 64) ? '.' : etoa64[e-64];
      /* When changing this update definition of LLEN above. */
      l[11 + (grplen * cols - 1)/octspergrp + p] =
#ifdef __MVS__
      (e >= 64)
#else
      (e > 31 && e < 127)
#endif
      ? e : '.';
      if (e)
    nonzero++;
      n++;
      if (++p == cols)
    {
      l[c = (11 + (grplen * cols - 1)/octspergrp + p)] = '\n'; l[++c] = '\0';
      xxdline(fpo, l, autoskip ? nonzero : 1);
      nonzero = 0;
      p = 0;
    }
    }
  if (p)
    {
      l[c = (11 + (grplen * cols - 1)/octspergrp + p)] = '\n'; l[++c] = '\0';
      xxdline(fpo, l, 1);
    }
  else if (autoskip)
    xxdline(fpo, l, -1);	/* last chance to flush out suppressed lines */

  fclose(fp);
  fclose(fpo);
  return 0;
}
