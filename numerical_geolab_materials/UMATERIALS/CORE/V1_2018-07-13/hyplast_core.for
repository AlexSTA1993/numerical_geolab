C     ALGORITHMS AND CODE CREATED BY IOANNIS STEFANOU, 2016, PARIS
C     ALL RIGHTS RESERVED
C
      SUBROUTINE USERMATMEL(STRESS,DE,DSDE,NSTR,PROPS,NPROPS,
     1 SVARSGP,NSVARSGP,NILL)
C
C      INCLUDE 'ABA_PARAM.INC'
C
      IMPLICIT NONE
C
      DOUBLE PRECISION STRESS(NSTR),DE(NSTR),DSDE(NSTR,NSTR),
     1  STRESS_T(NSTR),STRESS_TDT(NSTR),
     2  PROPS(NPROPS),SVARSGP(NSVARSGP)
      DOUBLE PRECISION DEL(NSTR,NSTR),DSTRESS(NSTR),RQ(0)
C
      INTEGER NPROPS,NSVARSGP,NILL,NSTR     
C
      STRESS_T=STRESS
      CALL GETELMATRIX(DEL,STRESS_T,NSTR,RQ,0,PROPS,NPROPS)
      DSTRESS=MATMUL(DEL,DE)         
      STRESS_TDT=STRESS_T+DSTRESS
      STRESS=STRESS_TDT
      SVARSGP(1:NSTR)=STRESS
      SVARSGP(NSTR+1:2*NSTR)=SVARSGP(NSTR+1:2*NSTR)+DE
      DSDE=DEL
      RETURN
      END
C      
      SUBROUTINE USERMATMPL(STRESS,DE,DSDE,NSTR,PROPS,NPROPS,
     1 NF,NH,SVARSGP,NSVARSGP,DTIME,NILL)
C
C      INCLUDE 'ABA_PARAM.INC'
C
      IMPLICIT NONE
C
      DOUBLE PRECISION STRESS(NSTR),DE(NSTR),DSDE(NSTR,NSTR),
     1  STRESS_T(NSTR),STRESS_TDT(NSTR),
     2  PROPS(NPROPS),SVARSGP(NSVARSGP)
      DOUBLE PRECISION RJAC(NF+NSTR+NH,NF+NSTR+NH),
     1  RIJAC(NF+NSTR+NH,NF+NSTR+NH),RB(NF+NSTR+NH),
     2  RDX(NF+NSTR+NH),RLAMDA(NF),!,DELASTIC(NSTR)
     2  RALAMDA(NF),STRESS_TR(NSTR),!,DSTRESS(NSTR)
     3  RQ(NH),RQ0(NH)
      DOUBLE PRECISION DEL(NSTR,NSTR),RAFST(NF,NSTR),RAFS(NSTR,NF),
     1  RAGS(NSTR,NF),RAFSS(NSTR,NSTR,NF),RAGSS(NSTR,NSTR,NF),
     2  RAF(NF),RF(NF),RFS(NSTR,NF),RGS(NSTR,NF),
     3  RFSS(NSTR,NSTR,NF),RGSS(NSTR,NSTR,NF),
     4  RID(NSTR,NSTR),RIDEL(NSTR,NSTR),
     5  DEPL(NSTR),RHQ(NH,NF),RHQS(NH,NSTR,NF),RHQQ(NH,NH,NF),
     7  RFQ(NH,NF),RAFQ(NH,NF),RAHQ(NH,NF),RAHQS(NH,NSTR,NF),
     8  RAHQQ(NH,NH,NF),RIHD(NH,NH),RGQS(NH,NSTR,NF),RAGQS(NH,NSTR,NF),
     9  RVISC(NF),RAVISC(NF)
      DOUBLE PRECISION RVTMP(NF+NSTR+NH),RVTMP2(NF+NSTR+NH),
     1  RMTMP1(NF+NSTR+NH,NF+NSTR+NH),
     2  RMTMP2(NF+NSTR+NH,NF+NSTR+NH),RMTMP3(NF+NSTR+NH,NF+NSTR+NH)
      DOUBLE PRECISION  RRRTMP1(NSTR,NH),RRRTMP2(NSTR,NSTR),
     1  RRRTMP3(NSTR,NSTR),RRRTMP4(NH,NH),RRRTMP5(NH,NH)
C     2  RRRTMP6(NA,NA),RRRTMP7(NA,NA)
      INTEGER NORDER(NF),NORDERINI(NF),NINVORDER(NF)
      DOUBLE PRECISION ELSTRAIN_T(NSTR)
C
      INTEGER NPROPS,NSVARSGP,NF,NH,NILL
      INTEGER ITER,NSITER,I,NA,NSTR
      DOUBLE PRECISION RTOL,RCRIT1,RCRIT2,RCRIT3,RCRIT4,RSIGN,RND,DTIME
	  DOUBLE PRECISION ratio
      RTOL=1.D-6 ! SET COVERGENCE TOLERANCE - IT IS ENOUGH TO SET IT ONLY HERE
	  
	  IF (DTIME.EQ.0.D0) DTIME=RTOL**2
	  
      NILL=0
      NSITER=0
      CALL GETID(RID,NSTR)
      IF (NH.GT.0) THEN
        RQ=SVARSGP(2*NSTR+1:2*NSTR+NH)
        RQ0=RQ
      END IF
      RALAMDA=0.D0
      RLAMDA=0.D0
      CALL GETVISCPARAMS(RVISC,NF,PROPS,NPROPS)
C     
      STRESS_T=STRESS
C
      DO I=1, NF
        NORDER(I)=I
        NORDERINI(I)=I
      END DO
C     CALCULATE TRIAL STRESS/ELASTICITY
      
      CALL GETELSTRAIN(ELSTRAIN_T,
     1   STRESS_T,NSTR,RQ0,NH,PROPS,NPROPS)
      RVTMP(1:NSTR)=ELSTRAIN_T+DE
      CALL GETELSTRESS(STRESS_TDT,
     1   RVTMP(1:NSTR),NSTR,RQ0,NH,PROPS,NPROPS)
      CALL GETELMATRIX(DEL,STRESS_T,NSTR,RQ0,NH,PROPS,NPROPS)
      
      IF (NF.EQ.0) THEN
        DEPL=0.D0
        RF=0.D0
        DSDE=DEL
        GO TO 40
      END IF
C
      CALL PSEUDOINVERSELA(DEL,RIDEL,NSTR,NSTR,NILL)
      STRESS_TR=STRESS_TDT      
C
      CALL CALCSURF(RF,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,NPROPS,NORDER)
50    CALL CALCGRADSURF(RFS,RGS,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,
     1  NPROPS,NORDER)
      CALL CALCSECGRADSURF(RFSS,RGSS,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,
     1  NPROPS,NORDER)
      IF (NH.NE.0) THEN
        CALL CALCHARDPARAMS(RHQ,RFQ,RHQQ,RHQS,RGQS,RQ,STRESS_TDT,
     1      NH,NSTR,NF,PROPS,NPROPS,NORDER)
      END IF
C
      CALL REFINESURF(RF,NF,STRESS_TDT,NSTR,PROPS,NPROPS,
     1  RAF,RFS,RGS,RFSS,RGSS,RVISC,RAFS,RAGS,RAFSS,RAGSS,RAVISC,
     2  RLAMDA,RALAMDA,RHQ,RFQ,RHQS,RHQQ,RAHQ,RAFQ,RAHQS,RAHQQ,
     3  RGQS,RAGQS,NH,NORDER,NA)
     
      IF (NA.EQ.0) THEN 
        DEPL=0.D0
        DSDE=DEL
        GO TO 40
      END IF
C     ITERATOR
30    ITER=1
      RDX=0.D0
      DO WHILE (ITER.GE.1)
        
        RJAC=0.D0
        RAFST(1:NA,:)=TRANSPOSE(RAFS(:,1:NA))
        DO I=1,NSTR
            RJAC(I,1:NSTR)=MATMUL(RAGSS(I,1:NSTR,1:NA),RLAMDA(1:NA))   
        END DO
        RJAC(1:NSTR,1:NSTR)=RIDEL+RJAC(1:NSTR,1:NSTR)
        RJAC(1:NSTR,NSTR+1:NSTR+NA)=RAGS(:,1:NA)
        RJAC(NSTR+1:NSTR+NA,1:NSTR)=RAFST(1:NA,:)
        !ADD VISCOPLASTIC TERMS
        DO I=1,NA
            RJAC(NSTR+I,NSTR+I)=RAVISC(I)/DTIME
        END DO

        RB=0.D0

!       THIS WAS FOR LINEAR ELASTICITY
!        RVTMP(1:NSTR)=STRESS_TR-STRESS_TDT
!        RB(1:NSTR)=MATMUL(RIDEL,RVTMP(1:NSTR))
!        RVTMP(1:NSTR)=MATMUL(RAGS(:,1:NA),RLAMDA(1:NA))
!        RB(1:NSTR)=RB(1:NSTR)-RVTMP(1:NSTR)
!        RB(NSTR+1:NSTR+NA)=-RAF(1:NA)

C     CALCULATE RESIDUAL FOR NON-LINEAR ELASTICITY
       CALL GETELSTRAIN(RVTMP(1:NSTR),
     1   STRESS_TDT,NSTR,RQ,NH,PROPS,NPROPS)
       RVTMP2(1:NSTR)=ELSTRAIN_T
       RB(1:NSTR)=DE-RVTMP(1:NSTR)+RVTMP2(1:NSTR)
       RVTMP(1:NSTR)=+MATMUL(RAGS(:,1:NA),RLAMDA(1:NA))
       RB(1:NSTR)=RB(1:NSTR)-RVTMP(1:NSTR)
       !FAILURE SURFACE
       RB(NSTR+1:NSTR+NA)=-RAF(1:NA)
       !VISCOPLASTIC UPDATE
       DO I=1,NA
        RVTMP(I)=RAVISC(I)*RLAMDA(I)/DTIME
       END DO
       RB(NSTR+1:NSTR+NA)=RB(NSTR+1:NSTR+NA)+RVTMP(1:NA)
        
        IF (NH.NE.0) THEN
            !FIRST DERIVATIVES
            !LINES 1 - NSTR
            RRRTMP1=RMTMP1(1:NSTR,1:NH)
            CALL GETDEELDB(RRRTMP1,
     1          STRESS_TDT,NSTR,RQ,NH,PROPS,NPROPS)
            RMTMP1(1:NSTR,1:NH)=RRRTMP1
            RJAC(1:NSTR,NSTR+NA+1:NSTR+NA+NH)=RMTMP1(1:NSTR,1:NH)
            !LINES NSTR+1 - NSTR+NA
            RJAC(NSTR+1:NSTR+NA,NSTR+NA+1:NSTR+NA+NH)=
     1          TRANSPOSE(RAFQ(:,1:NA))
            !LINES NSTR+NA+1 - NSTR+NA+NH
            RJAC(NA+NSTR+1:NA+NSTR+NH,NSTR+1:NSTR+NA)=RAHQ(:,1:NA)
            !SECOND DERIVATIVES
            !LINES 1 - NSTR
            DO I=1,NSTR
                RVTMP(1:NH)=MATMUL(RAGQS(1:NH,I,1:NA),RLAMDA(1:NA))
                RJAC(I,NA+NSTR+1:NA+NSTR+NH)=
     1              RJAC(I,NA+NSTR+1:NA+NSTR+NH)+RVTMP(1:NH)
            END DO
            !LINES NSTR+NA+1 - NSTR+NA+NH
            DO I=1,NH
                RVTMP(1:NSTR)=MATMUL(RAHQS(I,1:NSTR,1:NA),RLAMDA(1:NA)) 
                RJAC(NSTR+NA+I,1:NSTR)=
     1              RJAC(NSTR+NA+I,1:NSTR)+RVTMP(1:NSTR) 
                RVTMP(1:NH)=MATMUL(RAHQQ(I,1:NH,1:NA),RLAMDA(1:NA))
                RJAC(NSTR+NA+I,NSTR+NA+1:NSTR+NA+NH)=
     1              RJAC(NSTR+NA+I,NSTR+NA+1:NSTR+NA+NH)+RVTMP(1:NH)
            END DO            
            CALL GETID(RIHD,NH)
            RJAC(NSTR+NA+1:NSTR+NA+NH,NSTR+NA+1:NSTR+NA+NH)=-RIHD+
     1          RJAC(NSTR+NA+1:NSTR+NA+NH,NSTR+NA+1:NSTR+NA+NH)
            RVTMP(1:NH)=MATMUL(RAHQ(1:NH,1:NA),RLAMDA(1:NA))
            RB(NSTR+NA+1:NSTR+NA+NH)=RQ(1:NH)-RQ0(1:NH)-RVTMP(1:NH)    
        END IF
        !CALCULATE D_SOLUTION (MATRIX INVERSION AND MULTIPLICATION)
        CALL PSEUDOINVERSELA(RJAC(1:NA+NSTR+NH,1:NA+NSTR+NH),
     1      RIJAC(1:NA+NSTR+NH,1:NA+NSTR+NH),NA+NSTR+NH,NA+NSTR+NH,NILL)
        RDX(1:NA+NSTR+NH)=MATMUL(RIJAC(1:NA+NSTR+NH,1:NA+NSTR+NH),
     1      RB(1:NA+NSTR+NH))
        STRESS_TDT=STRESS_TDT+RDX(1:NSTR)
        RLAMDA(1:NA)=RDX(NSTR+1:NA+NSTR)+RLAMDA(1:NA)
        IF (NH.NE.0) THEN
            RQ(1:NH)=RQ(1:NH)+RDX(NA+NSTR+1:NA+NSTR+NH)
        END IF
        CALL GETELMATRIX(DEL,STRESS_TDT,NSTR,RQ,NH,PROPS,NPROPS)
        CALL PSEUDOINVERSELA(DEL,RIDEL,NSTR,NSTR,NILL)
        CALL CALCSURF(RAF,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,NPROPS,NORDER)
        CALL CALCGRADSURF(RAFS,RAGS,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,
     1          NPROPS,NORDER)
        CALL CALCSECGRADSURF(RAFSS,RAGSS,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,
     1          NPROPS,NORDER)
        IF (NH.NE.0) THEN
            CALL CALCHARDPARAMS(RAHQ,RAFQ,RAHQQ,RAHQS,RAGQS,RQ,
     1          STRESS_TDT,NH,NSTR,NF,PROPS,NPROPS,NORDER)
        END IF
        CALL GETNORM(RAF(1:NA),NA,RCRIT1)      
        CALL GETNORM(RDX(1:NSTR),NSTR,RCRIT2)
        CALL GETNORM(RDX(NSTR+1:NSTR+NA),NA,RCRIT3)
        IF (NH.NE.0) THEN
            CALL GETNORM(RDX(NSTR+NA+1:NSTR+NA+NH),NH,RCRIT4)
        ELSE
            RCRIT4=0.D0
        END IF
        IF (NA.LE.1) THEN
            IF ((RCRIT1.LE.RTOL).AND.(RCRIT2.LE.RTOL).AND. 
     1                      (RCRIT3.LE.RTOL).AND.(RCRIT4.LE.RTOL)) THEN
                ITER=0
            ELSE
                ITER=ITER+1
            END IF
        ELSE
            IF ((RCRIT2.LE.RTOL).AND. 
     1                      (RCRIT3.LE.RTOL).AND.(RCRIT4.LE.RTOL)) THEN
                ITER=0
            ELSE
                ITER=ITER+1
            END IF
        END IF
        !IF ((RCRIT3.GT.1.D6).OR.(ITER.GE.100).OR.(NILL.EQ.1)) THEN
        IF ((ITER.GE.100).OR.(NILL.EQ.1)) THEN
            !DO I=1,NSTR
            !    CALL RANDOM_NUMBER(RND)
            !    STRESS_TDT(I)=RND*STRESS_TR(I)
            !END DO
            !RLAMDA=0.D0
            !WRITE(6,*) '>UEL ERROR: MULTISURFACE PLASTICITY ALGORITHM
     1      !    FAILED TO CONVERGENCE (IN ITERATOR). 
     2      !    CHANGING INITIAL GUESS'
            !GO TO 30
            NILL=1
            RETURN
        END IF
      END DO
      
      NSITER=NSITER+1
      
      IF (NSITER.GE.50) THEN
          WRITE(6,*) '>UEL ERROR: MULTISURFACE PLASTICITY ALOGRITHM
     1          FAILED TO CONVERGENCE (IN ACTIVE SURFACES UPDATING).
     2          CHANGING INITIAL GUESS'
            DO I=1,NSTR
                CALL RANDOM_NUMBER(RND)
                STRESS_TDT(I)=RND*STRESS_TR(I)
            END DO
            RLAMDA=0.D0
            NSITER=0
            NORDER=NORDERINI
            CALL CALCSURF(RF,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,NPROPS,
     1          NORDER)
            GO TO 50
            RETURN
      END IF
C     CHECK SIGN OF LAGRANGE MULTIPLIERS
      CALL CHECKSIGN(RLAMDA(1:NA),NA,RSIGN,RTOL)
      IF ((RSIGN.EQ.-1).OR.(RCRIT1.GT.RTOL)) THEN 
C       RE-SET ACTIVE SURFACES
        IF (RSIGN.EQ.-1) THEN
            RF(1:NA)=RLAMDA(1:NA)
            CALL INVNORDER(NORDER,NINVORDER,NF)           
            IF (NA.LT.NF) THEN 
               CALL CALCSURF(RAF,NF,STRESS_TDT,NSTR,RQ,NH,
     1          PROPS,NPROPS,NORDER)
               RF(NA+1:NF)=RAF(NA+1:NF)
            END IF
        ELSE
            NINVORDER=NORDERINI
            CALL CALCSURF(RF,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,
     1          NPROPS,NINVORDER)
        ENDIF
        CALL REFINESURF(RF,NF,STRESS_TDT,NSTR,PROPS,NPROPS,
     1      RAF,RFS,RGS,RFSS,RGSS,RVISC,RAFS,RAGS,RAFSS,RAGSS,RAVISC,
     2      RLAMDA,RALAMDA,RHQ,RFQ,RHQS,RHQQ,RAHQ,RAFQ,RAHQS,RAHQQ,
     3      RGQS,RAGQS,NH,NINVORDER,NA)
        RLAMDA=RALAMDA
        NORDER=NINVORDER
        CALL CALCSURF(RAF,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,NPROPS,NORDER)
        CALL CALCGRADSURF(RAFS,RAGS,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,
     1      NPROPS,NORDER)
        CALL CALCSECGRADSURF(RAFSS,RAGSS,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,
     1      NPROPS,NORDER)
        IF (NH.NE.0) THEN
            CALL CALCHARDPARAMS(RAHQ,RAFQ,RAHQQ,RAHQS,RAGQS,RQ,
     1          STRESS_TDT,NH,NSTR,NF,PROPS,NPROPS,NORDER)
        END IF
        IF (NA.EQ.0) THEN
            NILL=1
            RETURN
        END IF
        GO TO 30
      ELSE
C     CONVERGED
C     CALCULATE PLASTIC STRAINS
        CALL CALCGRADSURF(RAFS,RAGS,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,
     1      NPROPS,NORDER)
        DEPL=MATMUL(RAGS(1:NSTR,1:NA),RLAMDA(1:NA))
C     CALCULATE CONSISTENT DSDE
        CALL CALCSECGRADSURF(RAFSS,RAGSS,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,
     1      NPROPS,NORDER)
        !determine � (�=RJAC(1:NSTR,1:NSTR))
        DO I=1,NSTR
            RJAC(I,1:NSTR)=MATMUL(RAGSS(I,1:NSTR,1:NA),RLAMDA(1:NA))
        END DO
        RJAC(1:NSTR,1:NSTR)=RIDEL+RJAC(1:NSTR,1:NSTR)
        !inverse � (�^-1=RIJAC(1:NSTR,1:NSTR))
        !RJAC is no longer needed but RIJAC it does
        RRRTMP2=RJAC(1:NSTR,1:NSTR)
        RRRTMP3=RIJAC(1:NSTR,1:NSTR)
        CALL PSEUDOINVERSELA(RRRTMP2,RRRTMP3,
     1      NSTR,NSTR,NILL)    !TO CHANGE WITH STH LIGHTER
        RJAC(1:NSTR,1:NSTR)=RRRTMP2
        RIJAC(1:NSTR,1:NSTR)=RRRTMP3
        !***do not replace RIJAC until ��/�� is calculated***
        !determine (�f/��)T (�ranspose)
        RAFST(1:NA,:)=TRANSPOSE(RAFS(:,1:NA))
        !determine (�f/��)T * �^-1 = RMTMP1(1:NA,1:NSTR)  
        RMTMP1(1:NA,1:NSTR)=    
     1   MATMUL(RAFST(1:NA,1:NSTR),RIJAC(1:NSTR,1:NSTR))
        !***do not replace RMTMP1 until ��/�� is calculated***
        !determine (�f/��)T * �^-1 * �g/�� = RMTMP3(1:NA,1:NA)
        RMTMP3(1:NA,1:NA)=MATMUL(RMTMP1(1:NA,1:NSTR),RAGS(1:NSTR,1:NA))
        !this is the M matrix
        !update M matrix for hardening before inversing
        IF (NH.NE.0) THEN 
            CALL CALCHARDPARAMS(RAHQ,RAFQ,RAHQQ,RAHQS,RAGQS,RQ,
     1          STRESS_TDT,NH,NSTR,NF,PROPS,NPROPS,NORDER)
            !determine �hq/�q*�
            DO I=1,NH
                RJAC(I,1:NH)=MATMUL(RAHQQ(I,1:NH,1:NA),RLAMDA(1:NA)) !RLAMDA and not RALAMDA is OK here
            END DO
            !determine � = I - �hq/�q*�
            RJAC(1:NH,1:NH)=RIHD-RJAC(1:NH,1:NH)
            !determine �^-1
            RRRTMP4=RJAC(1:NH,1:NH)
            RRRTMP5=RMTMP2(1:NH,1:NH)
            CALL PSEUDOINVERSELA(RRRTMP4,RRRTMP5,
     1          NH,NH,NILL) !TO CHANGE WITH STH LIGHTER
            RJAC(1:NH,1:NH)=RRRTMP4
            RMTMP2(1:NH,1:NH)=RRRTMP5
            !determine (�f/�q)T * �^-1
            RMTMP2(1:NA,1:NH)=
     1          MATMUL(TRANSPOSE(RAFQ(1:NH,1:NA)),RMTMP2(1:NH,1:NH))
            !determine (�f/�q)T * �^-1 * hq
            RMTMP2(1:NA,1:NA)=MATMUL(RMTMP2(1:NA,1:NH),RAHQ(1:NH,1:NA))
            !determine M = -(�f/�q)T * �^-1 * hq   +   (�f/��)T * �^-1 * �g/��
            RMTMP3(1:NA,1:NA)=-RMTMP2(1:NA,1:NA)+RMTMP3(1:NA,1:NA)
            RMTMP2(1:NA,1:NA)=0.D0
        END IF
        !inverse M matrix
C        RRRTMP6=RMTMP3(1:NA,1:NA)
C        RRRTMP7=RMTMP2(1:NA,1:NA)
        CALL PSEUDOINVERSELA(RMTMP3(1:NA,1:NA),RMTMP2(1:NA,1:NA),
     1      NA,NA,NILL)   !TO CHANGE WITH STH LIGHTER
C        RMTMP3(1:NA,1:NA)=RRRTMP6
C        RMTMP2(1:NA,1:NA)=RRRTMP7
        !determine M^-1 * (�f/��)T * �^-1
        RMTMP3(1:NA,1:NSTR)=
     1      MATMUL(RMTMP2(1:NA,1:NA),RMTMP1(1:NA,1:NSTR))
        !determine �g/�� * M^-1 * (�f/��)T * �^-1
        RMTMP1(1:NSTR,1:NSTR)=
     1      MATMUL(RAGS(1:NSTR,1:NA),RMTMP3(1:NA,1:NSTR))
        !determine I  -  �g/�� * M^-1 * (�f/��)T * �^-1
        RMTMP1(1:NSTR,1:NSTR)=RID-RMTMP1(1:NSTR,1:NSTR)
        !determine �^-1 * (I  -  �g/�� * M^-1 * (�f/��)T * �^-1)
        DSDE=MATMUL(RIJAC(1:NSTR,1:NSTR),RMTMP1(1:NSTR,1:NSTR))
        !this is the ��/��
        ratio=0.d-0
        DSDE=ratio*DEL+(1.d0-ratio)*DSDE !add a small part of elasticity

C       IF HERE, THE CALCULATION HAS FINISHED SUCCESSFULLY
C       VERIFY THAT ALL F ARE NEGATIVE AND/OR ZERO - IN THE DOMAIN
C       THIS IS DONE TO ASSURE MECHANICS ARE CORRECTLY SOLVED IN ANY CASE
        DO I=1,NF
            RAF(I)=RLAMDA(NORDER(I))
        END DO
        RLAMDA=RAF
        CALL CALCSURF(RF,NF,STRESS_TDT,NSTR,RQ,NH,PROPS,
     1      NPROPS,NORDERINI)
        CALL REFINESURF(RF,NF,STRESS_TDT,NSTR,PROPS,NPROPS,
     1      RAF,RFS,RGS,RFSS,RGSS,RVISC,RAFS,RAGS,RAFSS,RAGSS,RAVISC,
     2      RLAMDA,RALAMDA,RHQ,RFQ,RHQS,RHQQ,RAHQ,RAFQ,RAHQS,RAHQQ,
     4      RGQS,RAGQS,NH,NORDER,NA)
        RLAMDA=RALAMDA
        IF (NA.GE.1) THEN
            CALL GETNORM(RAF(1:NA),NA,RCRIT1) 
            IF (RCRIT1.GE.RTOL) THEN 
                GOTO 50
            END IF
        END IF
      END IF
C     FINAL OPERATIONS, SAVE TO SVARS (STATE VARIABLES)
40    STRESS=STRESS_TDT
      SVARSGP(1:NSTR)=STRESS
      SVARSGP(NSTR+1:2*NSTR)=SVARSGP(NSTR+1:2*NSTR)+DE
      IF (NH.NE.0) THEN
        SVARSGP(2*NSTR+1:2*NSTR+NH)=RQ
      END IF
      IF (NSVARSGP.GE.(3*NSTR+NH)) THEN
        SVARSGP(2*NSTR+1+NH:3*NSTR+NH)=SVARSGP(2*NSTR+1+NH:3*NSTR+NH)+
     1   DEPL
      END IF
      IF (NSVARSGP.GE.(3*NSTR+1+NH)) THEN
        SVARSGP(3*NSTR+1+NH:3*NSTR+1+NH)=NA
      END IF
      IF (NSVARSGP.GE.(3*NSTR+NF+1+NH)) THEN
        SVARSGP(3*NSTR+1+1+NH:3*NSTR+NF+1+NH)=RF
      END IF
      IF (NSVARSGP.GE.(3*NSTR+2*NF+1+NH)) THEN
        SVARSGP(3*NSTR+NF+1+1+NH:3*NSTR+2*NF+1+NH)=RLAMDA
      END IF
      IF (NSVARSGP.GE.(4*NSTR+2*NF+1+NH)) THEN
        SVARSGP(3*NSTR+2*NF+1+1+NH:4*NSTR+2*NF+1+NH)=
     1      SVARSGP(3*NSTR+2*NF+1+1+NH:4*NSTR+2*NF+1+NH)+DE-DEPL
      END IF
      RETURN
      END 
C
      SUBROUTINE INVNORDER(NORDER,NINVORDER,NF) 
C
C      INCLUDE 'ABA_PARAM.INC'
C
      IMPLICIT NONE
C
      INTEGER NORDER(NF),NINVORDER(NF)
      INTEGER NF,I
      DO I=1,NF
        NINVORDER(NORDER(I))=I
      END DO
      RETURN
      END
C
      SUBROUTINE REFINESURF(RF,NF,STRESSES,NSTR,PROPS,NPROPS,
     1  RAF,RFS,RGS,RFSS,RGSS,RVISC,RAFS,RAGS,RAFSS,RAGSS,RAVISC,
     2  RLAMDA,RALAMDA,RHQ,RFQ,RHQS,RHQQ,RAHQ,RAFQ,RAHQS,RAHQQ,
     3  RGQS,RAGQS,NH,MRSPM,NA)
C
C      INCLUDE 'ABA_PARAM.INC'
C
      IMPLICIT NONE
C
      DOUBLE PRECISION RF(NF),STRESSES(NSTR),PROPS(NPROPS),
     1  RFS(NSTR,NF),RGS(NSTR,NF),RFSS(NSTR,NSTR,NF), 
     2  RGSS(NSTR,NSTR,NF),RAF(NF),
     3  RAFS(NSTR,NF),RAGS(NSTR,NF),RAFSS(NSTR,NSTR,NF),  
     4  RAGSS(NSTR,NSTR,NF),RLAMDA(NF),RALAMDA(NF),
     5  RHQ(NH,NF),RHQS(NH,NSTR,NF),RHQQ(NH,NH,NF),
     6  RAHQ(NH,NF),RAHQS(NH,NSTR,NF),RAHQQ(NH,NH,NF),
     7  RFQ(NH,NF),RAFQ(NH,NF),RGQS(NH,NSTR,NF),RAGQS(NH,NSTR,NF),
     8  RVISC(NF),RAVISC(NF)
      INTEGER MRSPM(NF),NMRSPM(NF)
C
      INTEGER NF,NPROPS,I,NA,J,NNA,NSTR,NH
C
      NA=0
      NNA=NF+1
      DO I=1,NF
        J=MRSPM(I)
        IF (RF(J).GE.0.D0) THEN
            NA=NA+1
            RAF(NA)=RF(J)
            RAFS(:,NA)=RFS(:,J)
            RAGS(:,NA)=RGS(:,J)
            RAFSS(:,:,NA)=RFSS(:,:,J)
            RAGSS(:,:,NA)=RGSS(:,:,J)
            RAHQ(:,NA)=RHQ(:,J)
            RAFQ(:,NA)=RFQ(:,J)
            RAHQS(:,:,NA)=RHQS(:,:,J)
            RAHQQ(:,:,NA)=RHQQ(:,:,J)
            RAGQS(:,:,NA)=RGQS(:,:,J)
            RALAMDA(NA)=RLAMDA(J)
            RAVISC(NA)=RVISC(J)
            NMRSPM(J)=NA
        ELSE
            NNA=NNA-1
            RAF(NNA)=RF(J)
            RAFS(:,NNA)=RFS(:,J)
            RAGS(:,NNA)=RGS(:,J)
            RAFSS(:,:,NNA)=RFSS(:,:,J)
            RAGSS(:,:,NNA)=RGSS(:,:,J)
            RAHQ(:,NNA)=RHQ(:,J)
            RAFQ(:,NNA)=RFQ(:,J)
            RAHQS(:,:,NNA)=RHQS(:,:,J)
            RAHQQ(:,:,NNA)=RHQQ(:,:,J)
            RAGQS(:,:,NNA)=RGQS(:,:,J)
            RALAMDA(NNA)=RLAMDA(J)
            RAVISC(NNA)=RVISC(J)
            NMRSPM(J)=NNA
        END IF
      END DO
      MRSPM=NMRSPM
      RETURN
      END
C

*************************************************************************   
C     DEFINE VECTOR NORM
      SUBROUTINE GETNORM(VECTOR,M,RNORM)
C      
C      INCLUDE 'ABA_PARAM.INC'
C
C
      IMPLICIT NONE
C
      DOUBLE PRECISION SUM,RNORM
      DOUBLE PRECISION VECTOR(M)
      INTEGER I,M
      SUM=0.D0
      DO I=1,M
        SUM=SUM+VECTOR(I)**2
      ENDDO
      RNORM=DSQRT(SUM)
C
      RETURN
      END
C
*************************************************************************   
C     GET VECTOR ELEMENTS SIGN
      SUBROUTINE CHECKSIGN(VECTOR,M,RSIGN,RTOL)
C      
C      INCLUDE 'ABA_PARAM.INC'
C
C
      IMPLICIT NONE
C
      DOUBLE PRECISION RSIGN
      DOUBLE PRECISION VECTOR(M)
      INTEGER I,M
      DOUBLE PRECISION RTOL,RMTOL
      RMTOL=-RTOL
      RSIGN=+1.D0
      DO I=1,M
        IF (VECTOR(I).LT.RMTOL) THEN
            RSIGN=-1.D0
            GO TO 110
        END IF
      ENDDO
110   RETURN
      END 
*************************************************************************
*	SUBROUTINE FOR GENERAL MATRIX OPERATIONS                    	    *  
*************************************************************************
C     FORM IDENTITY MATRIX      
      SUBROUTINE GETID(RID,N)
C      
C      INCLUDE 'ABA_PARAM.INC'    
C
      IMPLICIT NONE
C      
      DOUBLE PRECISION RID(N,N)
      INTEGER N,K1
      RID=0.D0
      DO K1=1,N
        RID(K1,K1)=1.D0
      END DO  
      RETURN 
      END
C
      SUBROUTINE PSEUDOINVERSELA(A,RA,M,N,NILL)
C      
C      INCLUDE 'ABA_PARAM.INC'    
C
      IMPLICIT NONE
C  
!      INTERFACE 
!        SUBROUTINE DGESVD(r, Area)
!            REAL, INTENT(IN) :: r
!            REAL, INTENT(OUT) :: Area
!        END SUBROUTINE Compute_Area
!      END INTERFACE    
      
      INTEGER          M,N,LWMAX
      PARAMETER        (LWMAX=1000)
      INTEGER          INFO,LWORK,NILL,I
      DOUBLE PRECISION A(M,N),U(M,M),VT(N,N),S(N),
     $                 WORK(LWMAX)
      DOUBLE PRECISION RA(N,N),SP(N,N),RPA(M,N)
      DOUBLE PRECISION RSVDTOL
      PARAMETER (RSVDTOL=1.D-10)
*     .. External Subroutines ..
      EXTERNAL DGESVD
*     .. Intrinsic Functions ..
      INTRINSIC INT, MIN
      NILL=0
      RPA=A
      RA=0.D0
*     Query the optimal workspace.
      LWORK = -1
      CALL DGESVD( 'All', 'All', M, N, RPA, M, S, U, M, VT, N,
     $             WORK, LWORK, INFO )
      LWORK = MIN( LWMAX, INT( WORK( 1 ) ) )
*     Compute SVD.
      CALL DGESVD( 'All', 'All', M, N, RPA, M, S, U, M, VT, N,
     $             WORK, LWORK, INFO )
*     Check for convergence.
      IF( INFO.GT.0 ) THEN
         !WRITE(*,*)'The algorithm computing SVD failed to converge.'
         NILL=1
         RETURN
      END IF
      
      IF (M.NE.N) THEN 
        RETURN
      END IF
      
      SP=0.D0
      DO I=1,N
        IF (DABS(S(I)).GT.RSVDTOL) THEN
            SP(I,I)=S(I)**(-1)
        END IF
      END DO
      
      RA=MATMUL(MATMUL(TRANSPOSE(VT),SP),TRANSPOSE(U))
      
      RETURN
      END







