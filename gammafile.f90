!  program gammapack
!!
!  implicit none
!  integer :: nb=250
!  double precision, allocatable :: gamma(:,:,:)
!!
!  allocate(gamma(1:nb,1:nb,1:nb))
!  call findgamma(nb,gamma)
!!
!  open(10,file='gamma.dat', access='stream', status='replace')
!  write(10) gamma
!  close(10)
!  end program gammapack
!
!----------------------------------------------------------
!
  subroutine findgamma(nb,gamma)
!
  implicit none
  integer, intent(in) :: nb
  integer :: n, m, l
  double precision :: ci, si1, si2, si3, si4, ans
  double precision :: pi=3.141592653589793D0
  double precision, intent(out) :: gamma(1:nb,1:nb,1:nb)
!
  gamma = 0.0D0
  do n=1, nb
    do m=1, n
      do l=1, m
        call cisia((n+m+l)*pi,ci,si1)
        call cisia(dabs((n-m-l)*pi),ci,si2)
        if ((n-m-l).lt.0) si2=-si2
        call cisia(dabs((-n+m-l)*pi),ci,si3)
        if ((-n+m-l).lt.0) si3=-si3
        call cisia(dabs((-n-m+l)*pi),ci,si4)
        if ((-n-m+l).lt.0) si4=-si4
        ans = -0.5D0*dsqrt(2.0D0/3.0D0)*(si1+si2+si3+si4)
        gamma(n,m,l) = ans
        gamma(n,l,m) = ans
        gamma(m,n,l) = ans
        gamma(l,n,m) = ans
        gamma(m,l,n) = ans
        gamma(l,m,n) = ans
      enddo
    enddo
  enddo
  end subroutine findgamma
!
!******************************************************************
!*      Purpose: This program computes the cosine and sine        *
!*               integrals using subroutine CISIA                 *
!*      Input :  x  --- Argument of Ci(x) and Si(x)               *
!*      Output:  CI --- Ci(x)                                     *
!*               SI --- Si(x)                                     *
!*      Example:                                                  * 
!*                   x         Ci(x)          Si(x)               *
!*                ------------------------------------            *
!*                  0.0     - ì             .00000000             *
!*                  5.0     -.19002975     1.54993124             *
!*                 10.0     -.04545643     1.65834759             *
!*                 20.0      .04441982     1.54824170             * 
!*                 30.0     -.03303242     1.56675654             *
!*                 40.0      .01902001     1.58698512             *
!* -------------------------------------------------------------- *
!* REFERENCE: "Fortran Routines for Computation of Special        *
!*             Functions jin.ece.uiuc.edu/routines/routines.html" *
!*                                                                *
!*                              F90 Release By J-P Moreau, Paris. *
!*                                      (www.jpmoreau.fr)         *
!******************************************************************
!
        SUBROUTINE CISIA(X,CI,SI)

!       =============================================
!       Purpose: Compute cosine and sine integrals
!                Si(x) and Ci(x)  ( x ò 0 )
!       Input :  x  --- Argument of Ci(x) and Si(x)
!       Output:  CI --- Ci(x)
!                SI --- Si(x)
!       =============================================

        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        DIMENSION BJ(101)
        P2=1.570796326794897D0
        EL=.5772156649015329D0
        EPS=1.0D-15
        X2=X*X
        IF (X.EQ.0.0D0) THEN
           CI=-1.0D+300
           SI=0.0D0
        ELSE IF (X.LE.16.0D0) THEN
           XR=-.25D0*X2
           CI=EL+DLOG(X)+XR
           DO 10 K=2,40
              XR=-.5D0*XR*(K-1)/(K*K*(2*K-1))*X2
              CI=CI+XR
              IF (DABS(XR).LT.DABS(CI)*EPS) GO TO 15
10         CONTINUE
15         XR=X
           SI=X
           DO 20 K=1,40
              XR=-.5D0*XR*(2*K-1)/K/(4*K*K+4*K+1)*X2
              SI=SI+XR
              IF (DABS(XR).LT.DABS(SI)*EPS) RETURN
20         CONTINUE
        ELSE IF (X.LE.32.0D0) THEN
           M=INT(47.2+.82*X)
           XA1=0.0D0
           XA0=1.0D-100
           DO 25 K=M,1,-1
              XA=4.0D0*K*XA0/X-XA1
              BJ(K)=XA
              XA1=XA0
25            XA0=XA
           XS=BJ(1)
           DO 30 K=3,M,2
30            XS=XS+2.0D0*BJ(K)
           BJ(1)=BJ(1)/XS
           DO 35 K=2,M
35            BJ(K)=BJ(K)/XS
           XR=1.0D0
           XG1=BJ(1)
           DO 40 K=2,M
              XR=.25D0*XR*(2.0*K-3.0)**2/((K-1.0)*(2.0*K-1.0)**2)*X
40            XG1=XG1+BJ(K)*XR
           XR=1.0D0
           XG2=BJ(1)
           DO 45 K=2,M
              XR=.25D0*XR*(2.0*K-5.0)**2/((K-1.0)*(2.0*K-3.0)**2)*X
45            XG2=XG2+BJ(K)*XR
           XCS=DCOS(X/2.0D0)
           XSS=DSIN(X/2.0D0)
           CI=EL+DLOG(X)-X*XSS*XG1+2*XCS*XG2-2*XCS*XCS
           SI=X*XCS*XG1+2*XSS*XG2-DSIN(X)
        ELSE
           XR=1.0D0
           XF=1.0D0
           DO 50 K=1,9
              XR=-2.0D0*XR*K*(2*K-1)/X2
50            XF=XF+XR
           XR=1.0D0/X
           XG=XR
           DO 55 K=1,8
              XR=-2.0D0*XR*(2*K+1)*K/X2
55            XG=XG+XR
           CI=XF*DSIN(X)/X-XG*DCOS(X)/X
           SI=P2-XF*DCOS(X)/X-XG*DSIN(X)/X
        ENDIF
        RETURN
        END
!
