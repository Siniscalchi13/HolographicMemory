From Coq Require Import Reals.
Local Open Scope R_scope.

Module QMC_CHSH.
  (* Correlation model for the singlet state: E(a,b) = -cos(a - b). *)
  Definition E (a b:R) : R := - (cos (a - b)).

  Definition S (a ap b bp:R) : R := E a b + E a bp + E ap b - E ap bp.

  (* Tsirelson inequality upper bound (axiomatized for now). *)
  Axiom chsh_upper_bound : forall a ap b bp,
      Rabs (S a ap b bp) <= 2 * sqrt 2.

  (* Existence of settings that achieve the Tsirelson bound in magnitude. *)
  Axiom chsh_reaches_bound :
    Rabs (S 0 (PI/2) (PI/4) (- PI/4)) = 2 * sqrt 2.

End QMC_CHSH.

