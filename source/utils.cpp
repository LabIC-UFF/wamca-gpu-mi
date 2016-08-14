/**
 * @file   utils.cpp
 *
 * @brief  General auxiliary functions.
 *
 * @author Eyder Rios
 * @date   2011-09-12
 */


#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <netdb.h>
#include <stddef.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <iostream>
#include <fstream>
#include <string>
#include "utils.h"

using namespace std;

#define FNAME_CPUINFO       "/proc/cpuinfo"
#define TOKEN_BOGOMIPS      "bogomips"


/******************************************************************************************
 *
 *                                   FUNCTIONS
 *
 ******************************************************************************************/

uint
digits(llong n, uint base)
{
	uint digs = 0;
	do {
		digs++;
		n /= base;
	} while(n);

	return digs;
}

double
bogomips()
{
    ifstream  cpu(FNAME_CPUINFO);
    string    token;
    double    bogo;

    if(!cpu.is_open())
        return 0.0;

    bogo = 0.0;

    while(cpu >> token) {
        if(token == TOKEN_BOGOMIPS) {
            cpu >> token;
            cpu >> bogo;
            break;
        }
    }

    return bogo;
}

bool
getHostIPAddr(char *hostname, char *ip)
{
    sockaddr_in *h;
    addrinfo     hints,
                *servinfo;
    int          rv;
    bool         result;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ((rv = getaddrinfo(hostname,"http" , &hints , &servinfo)) != 0)
        return false;

    result = true;

    if(servinfo != NULL) {
        h = (sockaddr_in *) servinfo->ai_addr;
        strcpy(ip,inet_ntoa(h->sin_addr));
    }
    else {
        *ip = '\0';
        result = false;
    }

    freeaddrinfo(servinfo); // all done with this structure

    return result;
}

char *
stripext(char *path)
{
    int  i;

    for(i=strlen(path) - 1;i >= 0;i--) {
        if(path[i] == '.') {
            path[i] = '\0';
            break;
        }
    }

    return path;
}

char *
stripdir(char *path)
{
    int i;

    for(i=strlen(path) - 1;i >= 0;i--) {
        if(path[i] == '/')
            return path + i + 1;
    }
    return path;
}

char *
replaceext(char *path, char *ext)
{
    int  i;

    for(i=strlen(path) - 1;i >= 0;i--) {
        if(path[i] == '.') {
            if(ext)
                strcpy(path + i,ext);
            else
                path[i] = '\0';
            break;
        }
    }

    return path;
}

uint
bitCount(uint n)
{
  n = n - ((n >> 1) & 0x55555555);
  n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
  return (((n + (n >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
