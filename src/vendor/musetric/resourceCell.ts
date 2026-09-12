// Adapted from Musetric's MIT-licensed @musetric/utils package.
type ResourceController<Arg, Value> = {
  create: (arg: Arg) => Value;
  dispose: (value: Value) => void;
  equals: (currentArg: Arg, nextArg: Arg) => boolean;
};

export type ResourceCell<Arg, Value> = {
  get: (arg: Arg) => Value;
  dispose: () => void;
};

export const createResourceCell = <Arg, Value>(
  controller: ResourceController<Arg, Value>,
): ResourceCell<Arg, Value> => {
  let currentArg: Arg | undefined;
  let currentValue: Value | undefined;
  return {
    get(arg) {
      if (currentArg !== undefined && currentValue !== undefined && controller.equals(currentArg, arg)) {
        return currentValue;
      }
      if (currentValue !== undefined) controller.dispose(currentValue);
      currentArg = arg;
      currentValue = controller.create(arg);
      return currentValue;
    },
    dispose() {
      if (currentValue !== undefined) controller.dispose(currentValue);
      currentArg = undefined;
      currentValue = undefined;
    },
  };
};
