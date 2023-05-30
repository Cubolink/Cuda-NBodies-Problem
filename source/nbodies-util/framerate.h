/*****************************************

 framerate.h

*****************************************/

#if defined WIN32
#include <winsock.h>
#include <chrono>

int getTimeOfDay(struct timeval* tp, struct timezone* tzp) {
    // Thanks to: https://stackoverflow.com/questions/10905892/equivalent-of-gettimeofday-for-windows
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);
    time = ((uint64_t) file_time.dwLowDateTime);
    time += ((uint64_t) file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}

#else
#include <sys/time.h>
#endif

#include <stdio.h>

struct timeval frameStartTime, frameEndTime;
char	appTitle[64] = "OpenGL App"; // Window title
float	refreshtime = 1.0f; // FPS refresh period
float	gElapsedTime = 0.0f; // Current frame elapsed time
float	gTimeAccum = 0.0f; // Time accumulator for refresh
int		gFrames = 0; // Frame accumulator
float	gFPS = 0.0f; // Faramerate

void framerateTitle(char* title) {
	strcpy(appTitle, title);
	glutSetWindowTitle(title);
}

void framerateUpdate(void)
{
	getTimeOfDay(&frameEndTime, NULL);
	
	gElapsedTime = frameEndTime.tv_sec - frameStartTime.tv_sec + 
               ((frameEndTime.tv_usec - frameStartTime.tv_usec)/1.0E6);
  frameStartTime = frameEndTime;
  gTimeAccum += gElapsedTime;
  gFrames++;
  
	if (gTimeAccum > refreshtime)
	{  
		char title[64];
		gFPS = (float) gFrames / gTimeAccum;
		sprintf(title, "%s : %3.1f fps", appTitle, gFPS);
		glutSetWindowTitle(title);
		gTimeAccum = 0.0f;
		gFrames = 0;
	}
}
