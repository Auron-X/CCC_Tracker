#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h> 

void *func(void *arg) {
while(1)  
printf("1111111111\n");

}

int main()
{
    pthread_t thread;
    int id;
 if (id = pthread_create(&thread, NULL, func, NULL))
    {
    printf("ERROR!!!!!!!!!");
    return 0;
    } 
    
    while (1){
    printf("222222\n");
    
    }
    return 1;
}

